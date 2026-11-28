//! Space heater: the path tracer on the GPU through Vulkan compute.
//!
//! One shader invocation per character cell traces the scene
//! (`src/gpu/frame.wgsl`, compiled to SPIR-V by `build.rs`), and
//! [`GpuRenderer::render`] reads the text stream back for the CPU to draw
//! (per cell the glyph level, the solid hit, the display luminance and the
//! depth; the overlays need the last three).
//!
//! Vulkan 1.1 core, within what a Radeon HD 7850 (GCN 1.0) offers on AMD's
//! legacy driver or Mesa's RADV. Every floor pattern is on the GPU; a scene
//! with more than 32 solids stays on the CPU: see [`supports`].
//!
//! (The shader also has the Windows screensaver's discrete-pixel modes,
//! which present straight to its window; this build never selects them, but
//! their buffers stay bound so the pipeline layout matches the shader.)

use std::ffi::{CStr, c_void};

use ash::vk;

use crate::render::{FrameView, NO_HIT, Projection, RenderSettings};
use crate::scene::{Pattern, Scene};

/// The compiled shader (see `build.rs`).
static SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/frame.spv"));

/// Floor patterns the shader knows, by its pattern number: all of them.
const PATTERNS: [Pattern; 14] = [
    Pattern::Solid,
    Pattern::Checker,
    Pattern::Diamond,
    Pattern::Brick,
    Pattern::Triangles,
    Pattern::Hex,
    Pattern::Polka,
    Pattern::Radial,
    Pattern::Quasicrystal,
    Pattern::Rain,
    Pattern::Ocean,
    Pattern::Julia,
    Pattern::Clouds,
    Pattern::Carpet,
];

/// The most solids the shader takes (one bit each in its masks).
pub const MAX_SOLIDS: usize = 32;

/// Workgroup size per axis (`@workgroup_size(8, 8)`: 64 invocations, one
/// GCN wavefront).
const GROUP: u32 = 8;

/// Storage buffers: frame, solids, planes, cells, palette, pixels, overlay,
/// the floor pattern's items, the halo's lines, the last traced picture.
const BUFFERS: usize = 10;
const B_FRAME: usize = 0;
const B_SOLIDS: usize = 1;
const B_PLANES: usize = 2;
const B_CELLS: usize = 3;
const B_PALETTE: usize = 4;
const B_PIXELS: usize = 5;
const B_OVERLAY: usize = 6;
const B_WAVES: usize = 7;
const B_HALO: usize = 8;
const B_TRACED: usize = 9;

/// Whether the GPU can render `scene` (otherwise the CPU does).
pub fn supports(scene: &Scene) -> bool {
    scene.solids.len() <= MAX_SOLIDS && PATTERNS.contains(&scene.ground.shade().pattern)
}

/// `Frame` in the shader: vec4s throughout, so std430 and C agree.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FrameData {
    cam_pos: [f32; 4],
    cam_fwd: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    proj: [f32; 4],
    trace: [f32; 4],
    light: [f32; 4],
    shade: [f32; 4],
    centroid: [f32; 4],
    ground0: [f32; 4],
    ground1: [f32; 4],
    ground2: [f32; 4],
    ground3: [f32; 4],
    dims: [u32; 4],
    misc: [u32; 4],
    groundi: [u32; 4],
    pix: [u32; 4],
    pix2: [u32; 4],
    pix3: [i32; 4],
    ov0: [i32; 4],
    ov1: [i32; 4],
    ov_off: [u32; 4],
    pat: [f32; 4],
    halo: [f32; 4],
}


/// `Solid` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SolidData {
    centre: [f32; 4],
    params: [f32; 4],
    rot0: [f32; 4],
    rot1: [f32; 4],
    rot2: [f32; 4],
    planes: [u32; 4],
}

fn v4(v: glam::Vec3A, w: f32) -> [f32; 4] {
    [v.x, v.y, v.z, w]
}

/// A storage buffer: host-visible and mapped for its whole life, or
/// device-local (the pixels, which never leave the GPU).
struct Buffer {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    ptr: *mut c_void,
    size: u64,
}

impl Buffer {
    const NONE: Self = Self { buf: vk::Buffer::null(), mem: vk::DeviceMemory::null(), ptr: std::ptr::null_mut(), size: 0 };
}

/// The GPU renderer: one Vulkan device and compute pipeline, reused every frame.
pub struct GpuRenderer {
    /// Keeps the Vulkan loader loaded while the instance lives.
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    desc_pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    bufs: [Buffer; BUFFERS],
    /// The device's name, for logs.
    pub name: String,
    frame: u32,
    /// Seconds the last frame waited for the GPU (the CPU idle meanwhile).
    last_wait: f32,
    // the last frame, unpacked
    lum: Vec<f32>,
    glyphs: Vec<u8>,
    depth: Vec<f32>,
    ids: Vec<u8>,
    proj: Option<Projection>,
}

fn err(what: &str) -> impl Fn(vk::Result) -> String + '_ {
    move |e| format!("{what}: {e}")
}

impl GpuRenderer {
    /// Open the first Vulkan 1.1 device with a compute queue (a discrete GPU
    /// before an integrated one). `Err` when there is no Vulkan loader or
    /// no such device: the caller renders on the CPU.
    pub fn new() -> Result<Self, String> {
        // SAFETY: loading the system Vulkan loader; every call below follows
        // the Vulkan spec's valid usage for the objects it creates and only
        // uses handles created here.
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("no Vulkan loader: {e}"))?;
            let version = entry.try_enumerate_instance_version().map_err(err("instance version"))?.unwrap_or(vk::API_VERSION_1_0);
            if version < vk::API_VERSION_1_1 {
                return Err("Vulkan 1.1 is needed".into());
            }
            let app = vk::ApplicationInfo::default()
                .application_name(c"platomon")
                .engine_name(c"platomon space heater")
                .api_version(vk::API_VERSION_1_1);
            let instance =
                entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None).map_err(err("create instance"))?;
            match Self::with_instance(entry, instance.clone()) {
                Ok(r) => Ok(r),
                Err(e) => {
                    instance.destroy_instance(None);
                    Err(e)
                }
            }
        }
    }

    /// # Safety
    /// `instance` is live; on `Err` the caller destroys it.
    unsafe fn with_instance(entry: ash::Entry, instance: ash::Instance) -> Result<Self, String> {
        // SAFETY: see `new`.
        unsafe {
            let mut best: Option<(u32, vk::PhysicalDevice, u32, vk::PhysicalDeviceProperties)> = None;
            for pd in instance.enumerate_physical_devices().map_err(err("list devices"))? {
                let props = instance.get_physical_device_properties(pd);
                if props.api_version < vk::API_VERSION_1_1 {
                    continue;
                }
                let Some(family) = instance
                    .get_physical_device_queue_family_properties(pd)
                    .iter()
                    .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                else {
                    continue;
                };
                let rank = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                    _ => 0, // a software rasteriser last
                };
                if best.as_ref().is_none_or(|b| rank > b.0) {
                    best = Some((rank, pd, family as u32, props));
                }
            }
            let (_, pd, family, props) = best.ok_or("no Vulkan 1.1 device with a compute queue")?;
            let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned();
            let mem_props = instance.get_physical_device_memory_properties(pd);

            let prio = [1.0f32];
            let queue_info = vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&prio);
            let device = instance
                .create_device(pd, &vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info)), None)
                .map_err(err("create device"))?;
            let queue = device.get_device_queue(family, 0);

            let code = ash::util::read_spv(&mut std::io::Cursor::new(SPIRV)).map_err(|e| format!("shader: {e}"))?;
            let module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(err("shader module"))?;
            let bindings: Vec<_> = (0..BUFFERS as u32)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = device
                .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)
                .map_err(err("set layout"))?;
            let layout = device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)), None)
                .map_err(err("pipeline layout"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let pipeline = device
                .create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None)
                .map_err(|(_, e)| format!("compute pipeline: {e}"))?[0];
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: BUFFERS as u32 }];
            let desc_pool = device
                .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)
                .map_err(err("descriptor pool"))?;
            let set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(std::slice::from_ref(&set_layout)),
                )
                .map_err(err("descriptor set"))?[0];
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(err("command pool"))?;
            let cmd = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
                )
                .map_err(err("command buffer"))?[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(err("fence"))?;

            let mut me = Self {
                _entry: entry,
                instance,
                device,
                queue,
                mem_props,
                pool,
                cmd,
                fence,
                module,
                set_layout,
                layout,
                pipeline,
                desc_pool,
                set,
                bufs: [Buffer::NONE; BUFFERS],
                name,
                frame: 0,
                last_wait: 0.0,
                lum: Vec::new(),
                glyphs: Vec::new(),
                depth: Vec::new(),
                ids: Vec::new(),
                proj: None,
            };
            me.ensure(B_FRAME, size_of::<FrameData>() as u64)?;
            me.ensure(B_SOLIDS, (MAX_SOLIDS * size_of::<SolidData>()) as u64)?;
            me.ensure(B_PLANES, (MAX_SOLIDS * crate::scene::MAX_AXES * 2 * 16) as u64)?;
            me.ensure(B_CELLS, 16)?;
            me.ensure(B_PALETTE, 16)?;
            me.ensure(B_PIXELS, 16)?;
            me.ensure(B_OVERLAY, 16)?;
            me.ensure(B_WAVES, 16)?;
            me.ensure(B_HALO, 16)?;
            me.ensure(B_TRACED, 16)?;
            Ok(me)
        }
    }

    /// A memory type for `bits`: host-visible and coherent (cached
    /// preferred: read back every frame), or for the pixels device-local.
    fn memory_type(&self, bits: u32, device_local: bool) -> Option<u32> {
        let types = &self.mem_props.memory_types[..self.mem_props.memory_type_count as usize];
        let fits = |i: usize, f: vk::MemoryPropertyFlags| bits & (1 << i) != 0 && types[i].property_flags.contains(f);
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let order: &[vk::MemoryPropertyFlags] = if device_local {
            &[vk::MemoryPropertyFlags::DEVICE_LOCAL]
        } else {
            &[vk::MemoryPropertyFlags::from_raw(host.as_raw() | vk::MemoryPropertyFlags::HOST_CACHED.as_raw()), host]
        };
        order.iter().find_map(|&f| (0..types.len()).find(|&i| fits(i, f))).map(|i| i as u32)
    }

    /// Make buffer `k` at least `size` bytes (recreated and rebound if smaller).
    fn ensure(&mut self, k: usize, size: u64) -> Result<(), String> {
        if self.bufs[k].size >= size && self.bufs[k].buf != vk::Buffer::null() {
            return Ok(());
        }
        let size = size.max(256).next_power_of_two();
        let local = k == B_PIXELS || k == B_TRACED;
        // SAFETY: the device is idle between frames (every frame waits on
        // its fence), so the old buffer is no longer in use.
        unsafe {
            self.free(k);
            let d = &self.device;
            let usage = if local {
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST
            } else {
                vk::BufferUsageFlags::STORAGE_BUFFER
            };
            let buf = d
                .create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE), None)
                .map_err(err("create buffer"))?;
            let req = d.get_buffer_memory_requirements(buf);
            let Some(ty) = self.memory_type(req.memory_type_bits, local) else {
                d.destroy_buffer(buf, None);
                return Err("no suitable GPU memory".into());
            };
            let mem = match d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None) {
                Ok(m) => m,
                Err(e) => {
                    d.destroy_buffer(buf, None);
                    return Err(format!("allocate memory: {e}"));
                }
            };
            d.bind_buffer_memory(buf, mem, 0).map_err(err("bind memory"))?;
            let ptr = if local {
                std::ptr::null_mut()
            } else {
                d.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).map_err(err("map memory"))?
            };
            self.bufs[k] = Buffer { buf, mem, ptr, size };
            let info = [vk::DescriptorBufferInfo { buffer: buf, offset: 0, range: vk::WHOLE_SIZE }];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(k as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info);
            d.update_descriptor_sets(&[write], &[]);
        }
        Ok(())
    }

    /// # Safety
    /// The GPU is not using buffer `k`.
    unsafe fn free(&mut self, k: usize) {
        let b = &mut self.bufs[k];
        if b.buf != vk::Buffer::null() {
            // SAFETY: created by `ensure`, idle per the caller.
            unsafe {
                if !b.ptr.is_null() {
                    self.device.unmap_memory(b.mem);
                }
                self.device.destroy_buffer(b.buf, None);
                self.device.free_memory(b.mem, None);
            }
        }
        *b = Buffer::NONE;
    }

    /// Copy `data` to the start of host-visible buffer `k` (sized by
    /// [`Self::ensure`]).
    fn upload<T: Copy>(&self, k: usize, data: &[T]) {
        let bytes = std::mem::size_of_val(data);
        debug_assert!(bytes as u64 <= self.bufs[k].size && !self.bufs[k].ptr.is_null());
        // SAFETY: the mapping is at least `bytes` long and the GPU is idle;
        // `T` is plain `repr(C)` data.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, self.bufs[k].ptr as *mut u8, bytes) };
    }

    /// Upload one frame's scene for a `w × h` text-stream dispatch.
    fn prepare(&mut self, scene: &Scene, w: usize, h: usize, rs: &RenderSettings, exposure: f32, ramp_len: usize) -> Result<Projection, String> {
        let proj = Projection::new(scene.camera, w, h, rs);
        self.frame = self.frame.wrapping_add(1);
        let g = scene.ground.shade();
        let pattern = PATTERNS.iter().position(|p| *p == g.pattern).ok_or("floor pattern not on the GPU")? as u32;
        if scene.solids.len() > MAX_SOLIDS {
            return Err("too many solids for the GPU".into());
        }
        // solids and their face planes
        let mut solids = Vec::with_capacity(scene.solids.len());
        let mut planes: Vec<[f32; 4]> = Vec::with_capacity(scene.solids.len() * crate::scene::MAX_AXES);
        for s in &scene.solids {
            let (axes, offs, symmetric) = s.poly.planes();
            let first = planes.len() as u32;
            planes.extend(axes.iter().zip(offs).map(|(a, o)| v4(*a, *o)));
            let (curve, dish) = s.poly.curve_params();
            if curve < 0.0 {
                planes.extend(dish.iter().map(|&(cd, r)| [cd, r, 0.0, 0.0]));
            }
            let r = s.inv_rot;
            solids.push(SolidData {
                centre: v4(s.center, s.scale),
                params: [s.inv_scale, s.albedo, curve, 0.0],
                rot0: v4(r.x_axis, 0.0),
                rot1: v4(r.y_axis, 0.0),
                rot2: v4(r.z_axis, 0.0),
                planes: [first, axes.len() as u32, symmetric as u32, 0],
            });
        }
        if planes.is_empty() {
            planes.push([0.0; 4]);
        }
        if solids.is_empty() {
            solids.push(SolidData::default());
        }
        let (items, pat) = scene.ground.shader_pattern();
        let cam = scene.camera;
        let t = &rs.trace;
        let frame = FrameData {
            cam_pos: v4(cam.pos, 0.0),
            cam_fwd: v4(cam.forward, 0.0),
            cam_right: v4(cam.right, 0.0),
            cam_up: v4(cam.up, 0.0),
            proj: proj.shader_params(),
            trace: [t.max_dist, t.detail_dist.min(3.0e38), proj.pixel_angle(), scene.extent],
            light: v4(scene.light.center, scene.light.radius),
            shade: [scene.light.emission, scene.sky, scene.background, exposure],
            centroid: v4(scene.centroid, scene.slab_top),
            ground0: [g.base, g.contrast, g.inv_scale, g.fade_rate],
            ground1: [g.pivot.x, g.pivot.y, g.sin, g.cos],
            ground2: [g.offset.x, g.offset.y, g.shape.dot_size, g.shape.dot_spacing],
            ground3: [g.shape.dot_variability, 0.0, 0.0, 0.0],
            dims: [w as u32, h as u32, rs.spp.max(1), t.bounces],
            misc: [t.max_steps, scene.solids.len() as u32, self.frame, ramp_len.max(1) as u32],
            groundi: [pattern, g.enabled as u32, g.shape.invert_half as u32, 0],
            // text-stream mode: the pixel, overlay and halo fields stay zero
            pat,
            ..Default::default()
        };
        if !items.is_empty() {
            self.ensure(B_WAVES, std::mem::size_of_val(&items[..]) as u64)?;
            self.upload(B_WAVES, &items);
        }
        self.ensure(B_SOLIDS, std::mem::size_of_val(&solids[..]) as u64)?;
        self.ensure(B_PLANES, std::mem::size_of_val(&planes[..]) as u64)?;
        self.ensure(B_CELLS, (w * h * 8) as u64)?;
        self.upload(B_FRAME, &[frame]);
        self.upload(B_SOLIDS, &solids);
        self.upload(B_PLANES, &planes);
        Ok(proj)
    }

    /// Begin the frame's commands: bind and dispatch `w × h` invocations.
    ///
    /// # Safety
    /// The command buffer is idle (the last frame's fence was waited on).
    unsafe fn record_dispatch(&self, w: usize, h: usize) -> Result<(), String> {
        let (d, cb) = (&self.device, self.cmd);
        // SAFETY: per the caller.
        unsafe {
            d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()).map_err(err("reset commands"))?;
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .map_err(err("begin commands"))?;
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[self.set], &[]);
            d.cmd_dispatch(cb, (w as u32).div_ceil(GROUP), (h as u32).div_ceil(GROUP), 1);
        }
        Ok(())
    }

    /// Submit the recorded frame (waiting on / signalling the given
    /// semaphores) and wait for it.
    ///
    /// # Safety
    /// The command buffer is recorded and ended.
    unsafe fn submit_and_wait(&mut self, wait: Option<vk::Semaphore>, signal: Option<vk::Semaphore>) -> Result<(), String> {
        let d = &self.device;
        let waits: Vec<vk::Semaphore> = wait.into_iter().collect();
        let stages = [vk::PipelineStageFlags::TRANSFER];
        let signals: Vec<vk::Semaphore> = signal.into_iter().collect();
        let submit = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&self.cmd))
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stages[..waits.len()])
            .signal_semaphores(&signals);
        // SAFETY: per the caller; the fence is unsignalled (reset after every wait).
        unsafe {
            d.queue_submit(self.queue, &[submit], self.fence).map_err(err("submit"))?;
            let t = std::time::Instant::now();
            let waited = d.wait_for_fences(&[self.fence], true, 10_000_000_000);
            self.last_wait = t.elapsed().as_secs_f32();
            d.reset_fences(&[self.fence]).map_err(err("reset fence"))?;
            waited.map_err(err("wait for the GPU"))
        }
    }

    /// Render `scene` into a `w × h` text stream: glyph levels for a ramp of
    /// `ramp_len` glyphs at `exposure`. `Err` if the GPU failed (the caller
    /// falls back to the CPU); the scene must pass [`supports`].
    pub fn render(&mut self, scene: &Scene, w: usize, h: usize, rs: &RenderSettings, exposure: f32, ramp_len: usize) -> Result<(), String> {
        let n = w * h;
        if n == 0 {
            self.proj = Some(Projection::new(scene.camera, w, h, rs));
            self.lum.clear();
            self.glyphs.clear();
            self.depth.clear();
            self.ids.clear();
            return Ok(());
        }
        self.proj = Some(self.prepare(scene, w, h, rs, exposure, ramp_len)?);
        // SAFETY: valid handles; the last frame's fence was waited on.
        unsafe {
            self.record_dispatch(w, h)?;
            let (d, cb) = (&self.device, self.cmd);
            let barrier = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
            d.end_command_buffer(cb).map_err(err("end commands"))?;
            self.submit_and_wait(None, None)?;
        }

        // unpack the text stream
        // SAFETY: the shader wrote `n` cells of two u32s; the fence wait and
        // the host-read barrier make them visible.
        let cells = unsafe { std::slice::from_raw_parts(self.bufs[B_CELLS].ptr as *const [u32; 2], n) };
        self.lum.clear();
        self.glyphs.clear();
        self.depth.clear();
        self.ids.clear();
        self.lum.extend(cells.iter().map(|c| (c[0] >> 16) as f32 / 65535.0));
        self.glyphs.extend(cells.iter().map(|c| (c[0] & 0xff) as u8));
        self.ids.extend(cells.iter().map(|c| ((c[0] >> 8) & 0xff) as u8));
        self.depth.extend(cells.iter().map(|c| f32::from_bits(c[1])));
        Ok(())
    }

    /// Seconds the last frame spent waiting for the GPU to finish: not CPU
    /// work, so the CPU budget leaves it out.
    pub fn last_wait(&self) -> f32 {
        self.last_wait
    }

    /// The last text-stream frame: luminance, glyph levels and the
    /// overlays' depth and ids.
    pub fn view(&self) -> Option<FrameView<'_>> {
        Some(FrameView { radiance: &[], lum: Some(&self.lum), glyphs: Some(&self.glyphs), depth: &self.depth, ids: &self.ids, proj: self.proj? })
    }
}

impl Drop for GpuRenderer {
    fn drop(&mut self) {
        // SAFETY: every handle was created by this renderer; waiting for the
        // device first makes sure none is still in use.
        unsafe {
            let _ = self.device.device_wait_idle();
            for k in 0..self.bufs.len() {
                self.free(k);
            }
            let d = &self.device;
            d.destroy_fence(self.fence, None);
            d.destroy_command_pool(self.pool, None);
            d.destroy_descriptor_pool(self.desc_pool, None);
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
            d.destroy_shader_module(self.module, None);
            d.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// (the shader's NO_HIT and GROUND are the CPU's)
const _: () = assert!(NO_HIT == 254 && crate::render::march::GROUND == 255);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::Renderer;
    use crate::scene::{AnimParams, GroundParams};

    /// The GPU's picture is the CPU's, up to sampling noise (skipped where
    /// there is no Vulkan 1.1 GPU).
    #[test]
    fn gpu_matches_cpu() {
        let mut g = match GpuRenderer::new() {
            Ok(g) => g,
            Err(e) => return eprintln!("no GPU here ({e}): skipped"),
        };
        let ramp = 16;
        for pattern in PATTERNS {
            let mut scene = Scene::demo(&AnimParams::default()).with_ground(&GroundParams { pattern, lum: 6.0, contrast: 0.7, ..Default::default() });
            // the animated floors part-way through their motion
            scene.ground.advance(2.5);
            assert!(supports(&scene));
            let (w, h) = (80, 30);
            let rs = RenderSettings { spp: 48, ..RenderSettings::default() };
            g.render(&scene, w, h, &rs, 1.0, ramp).unwrap();
            let gv = g.view().unwrap();
            let mut cpu = Renderer::new();
            cpu.render(&scene, w, h, &rs);
            let mut cl = Vec::new();
            crate::ascii::tone_buffer(cpu.view().unwrap().radiance, 1.0, &mut cl);
            let gl = gv.lum.unwrap();
            let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
            assert!((mean(gl) - mean(&cl)).abs() < 0.01, "{pattern:?}: mean {} vs {}", mean(gl), mean(&cl));
            let close = gl.iter().zip(&cl).filter(|(a, b)| (*a - *b).abs() < 1.5 / ramp as f32).count();
            assert!(close as f32 > 0.97 * (w * h) as f32, "{pattern:?}: {close} of {} cells close", w * h);
            // the same surfaces hit, and the glyph levels are the luminance's
            let ids_same = gv.ids.iter().zip(cpu.view().unwrap().ids).filter(|(a, b)| a == b).count();
            assert!(ids_same as f32 > 0.99 * (w * h) as f32, "{pattern:?}: ids {ids_same}");
            for (&lv, &l) in gv.glyphs.unwrap().iter().zip(gl) {
                assert!((lv as i32 - ((l * ramp as f32) as i32).min(ramp as i32 - 1)).abs() <= 1);
            }
        }
    }

    #[test]
    fn curved_solids_match_cpu() {
        let mut g = match GpuRenderer::new() {
            Ok(g) => g,
            Err(e) => return eprintln!("no GPU here ({e}): skipped"),
        };
        let mut scene = Scene::demo(&AnimParams::default());
        for (i, c) in [-10.0, -5.0, 0.0, 5.0, 10.0].into_iter().enumerate() {
            scene.set_curvature(i, c);
        }
        let (w, h) = (96, 36);
        let rs = RenderSettings { spp: 48, ..RenderSettings::default() };
        g.render(&scene, w, h, &rs, 1.0, 16).unwrap();
        let gv = g.view().unwrap();
        let mut cpu = Renderer::new();
        cpu.render(&scene, w, h, &rs);
        let ids_same = gv.ids.iter().zip(cpu.view().unwrap().ids).filter(|(a, b)| a == b).count();
        assert!(ids_same as f32 > 0.99 * (w * h) as f32, "ids {ids_same} of {}", w * h);
        let mut cl = Vec::new();
        crate::ascii::tone_buffer(cpu.view().unwrap().radiance, 1.0, &mut cl);
        let close = gv.lum.unwrap().iter().zip(&cl).filter(|(a, b)| (*a - *b).abs() < 1.5 / 16.0).count();
        assert!(close as f32 > 0.97 * (w * h) as f32, "{close} of {} cells close", w * h);
    }
}
