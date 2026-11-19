//! The scene: one solid per slot on concentric rings, a ground plane at
//! y = 0, one spherical area light and a dim uniform sky.

pub mod camera;
pub mod sdf;

use glam::{Mat3A, Vec3A};
use std::f32::consts::{PI, TAU};

use crate::metrics::Channel;
pub use camera::{Camera, CameraParams, CameraRig};
use sdf::{Polyhedron, SolidKind, inverse_rotation};

/// Circumradius of every solid at scale 1.
pub const BASE_RADIUS: f32 = 1.0;
/// Radius of the innermost ring when it holds more than one solid.
pub const MIN_RING_RADIUS: f32 = 2.7;
/// Centre-to-centre spacing between neighbours and between rings.
pub const SPACING: f32 = 2.0 * BASE_RADIUS + 0.6;
/// Automatic placement fills a ring with this many solids before starting the next.
pub const RING_CAPACITY: usize = 8;
/// Gap between a solid's circumsphere and the ground.
pub const HOVER: f32 = 0.08;
/// Layout extent the light and camera were tuned for (default 5-solid ring).
const REF_EXTENT: f32 = MIN_RING_RADIUS + BASE_RADIUS;

/// Slightly different albedo per slot so neighbours separate in monochrome.
const ALBEDO: [f32; 8] = [0.85, 0.70, 0.62, 0.78, 0.66, 0.74, 0.58, 0.81];

/// Default slot list (the five classic resources).
pub const DEFAULT_SOLIDS: [SolidKind; 5] = [
    SolidKind::Icosahedron,
    SolidKind::Cube,
    SolidKind::Octahedron,
    SolidKind::Tetrahedron,
    SolidKind::Dodecahedron,
];

#[derive(Debug, Clone, Copy)]
pub struct Instance {
    pub poly: Polyhedron,
    pub center: Vec3A,
    /// Circumradius in world units.
    pub scale: f32,
    pub inv_scale: f32,
    pub inv_rot: Mat3A,
    pub albedo: f32,
}

impl Instance {
    /// `sdf(R⁻¹ · (p − c) / s) · s`, with `dc = p − c` precomputed.
    #[inline(always)]
    pub fn sdf_local(&self, dc: Vec3A) -> f32 {
        self.poly.eval(self.inv_rot * (dc * self.inv_scale)) * self.scale
    }

    #[inline(always)]
    pub fn sdf(&self, p: Vec3A) -> f32 {
        self.sdf_local(p - self.center)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Light {
    pub center: Vec3A,
    pub radius: f32,
    pub emission: f32,
}

/// Animation ranges from the config.
#[derive(Debug, Clone, Copy)]
pub struct AnimParams {
    pub min_scale: f32,
    pub max_scale: f32,
    pub idle_spin: f32,
    pub max_spin: f32,
}

impl Default for AnimParams {
    fn default() -> Self {
        Self { min_scale: 0.45, max_scale: 1.0, idle_spin: 0.15, max_spin: 3.0 }
    }
}

/// One solid to place.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotLayout {
    pub solid: SolidKind,
    /// Requested ring; `None` = automatic.
    pub ring: Option<u32>,
}

/// Ground-plane positions (y = 0) and the compacted ring index of every slot.
///
/// Explicit rings are honoured; automatic slots fill rings from the inside,
/// `RING_CAPACITY` per ring. Empty rings are skipped, so ring numbers only
/// order the rings. Within a ring, slots keep their `show` order, starting at
/// the front (+z); odd rings are rotated half a step so solids interleave.
pub fn ring_layout(rings: &[Option<u32>]) -> (Vec<Vec3A>, Vec<usize>) {
    let mut assigned: Vec<u32> = vec![0; rings.len()];
    let mut count = std::collections::BTreeMap::<u32, usize>::new();
    for r in rings.iter().flatten() {
        *count.entry(*r).or_default() += 1;
    }
    for (i, r) in rings.iter().enumerate() {
        assigned[i] = match r {
            Some(r) => *r,
            None => {
                let mut k = 0;
                while count.get(&k).copied().unwrap_or(0) >= RING_CAPACITY {
                    k += 1;
                }
                *count.entry(k).or_default() += 1;
                k
            }
        };
    }
    let order: Vec<u32> = count.keys().copied().collect();
    let compact: Vec<usize> = assigned.iter().map(|r| order.binary_search(r).unwrap_or(0)).collect();

    let mut radius_prev: Option<f32> = None;
    let mut radii = Vec::with_capacity(order.len());
    for k in 0..order.len() {
        let n = compact.iter().filter(|&&c| c == k).count();
        let chord = if n > 1 { SPACING / (2.0 * (PI / n as f32).sin()) } else { 0.0 };
        let r = match radius_prev {
            None if n == 1 => 0.0,
            None => chord.max(MIN_RING_RADIUS),
            Some(prev) => chord.max(prev + SPACING),
        };
        radii.push(r);
        radius_prev = Some(r);
    }

    let mut pos = vec![Vec3A::ZERO; rings.len()];
    for (k, &r) in radii.iter().enumerate() {
        let members: Vec<usize> = (0..rings.len()).filter(|&i| compact[i] == k).collect();
        let n = members.len() as f32;
        let offset = if k % 2 == 1 { PI / n } else { 0.0 };
        for (j, &i) in members.iter().enumerate() {
            let th = offset + j as f32 * TAU / n;
            pos[i] = Vec3A::new(r * th.sin(), 0.0, r * th.cos());
        }
    }
    (pos, compact)
}

#[derive(Debug, Clone)]
pub struct Scene {
    pub solids: Vec<Instance>,
    /// (pitch, yaw) per solid, radians.
    pub angles: Vec<(f32, f32)>,
    pub ring_pos: Vec<Vec3A>,
    pub ring_index: Vec<usize>,
    pub centroid: Vec3A,
    /// Distance from the centroid to the outermost solid's far edge.
    pub extent: f32,
    pub rig: CameraRig,
    pub camera: Camera,
    pub light: Light,
    /// Radiance of escaping secondary rays.
    pub sky: f32,
    /// Radiance of escaping camera rays.
    pub background: f32,
    pub ground_albedo: f32,
    /// Top of the y-slab that contains every solid.
    pub slab_top: f32,
}

impl Scene {
    pub fn new(slots: &[SlotLayout], anim: &AnimParams, cam: &CameraParams) -> Self {
        let rings: Vec<Option<u32>> = slots.iter().map(|s| s.ring).collect();
        let (ring_pos, ring_index) = ring_layout(&rings);
        let n = slots.len().max(1) as f32;
        let centroid = ring_pos.iter().copied().fold(Vec3A::ZERO, |a, b| a + b) / n;
        let extent = ring_pos.iter().map(|p| (*p - centroid).length()).fold(0.0, f32::max)
            + BASE_RADIUS * anim.max_scale;
        let k = extent / REF_EXTENT;

        let solids = slots
            .iter()
            .enumerate()
            .map(|(i, s)| Instance {
                poly: Polyhedron::new(s.solid),
                center: Vec3A::ZERO,
                scale: 1.0,
                inv_scale: 1.0,
                inv_rot: Mat3A::IDENTITY,
                albedo: ALBEDO[i % ALBEDO.len()],
            })
            .collect();
        // Stagger initial orientations so the idle scene isn't uniform.
        let angles = (0..slots.len()).map(|i| (0.4 + 0.3 * i as f32, 0.7 * i as f32)).collect();
        let rig = CameraRig::new(cam, centroid, extent);
        let mut s = Self {
            solids,
            angles,
            ring_pos,
            ring_index,
            centroid,
            extent,
            camera: rig.camera(),
            rig,
            light: Light {
                center: centroid + Vec3A::new(-7.0, 11.0, 4.0) * k,
                radius: 2.2 * k,
                emission: 90.0,
            },
            sky: 0.12,
            background: 0.0,
            ground_albedo: 0.15,
            slab_top: 2.0 * BASE_RADIUS + HOVER,
        };
        for i in 0..s.solids.len() {
            s.set_scale(i, anim.min_scale);
        }
        s.refresh_transforms();
        s
    }

    /// The classic five-solid scene with a static camera.
    pub fn demo(anim: &AnimParams) -> Self {
        let slots: Vec<SlotLayout> = DEFAULT_SOLIDS.iter().map(|&solid| SlotLayout { solid, ring: None }).collect();
        Self::new(&slots, anim, &CameraParams::default())
    }

    fn set_scale(&mut self, i: usize, scale: f32) {
        let r = (scale * BASE_RADIUS).max(1e-3);
        let inst = &mut self.solids[i];
        inst.scale = r;
        inst.inv_scale = 1.0 / r;
        // Rest just above the ground at every scale so contact shadows stay visible.
        inst.center = self.ring_pos[i] + Vec3A::new(0.0, r + HOVER, 0.0);
    }

    fn refresh_transforms(&mut self) {
        let mut top: f32 = 0.0;
        for (inst, &(pitch, yaw)) in self.solids.iter_mut().zip(&self.angles) {
            inst.inv_rot = inverse_rotation(pitch, yaw);
            top = top.max(inst.center.y + inst.scale);
        }
        self.slab_top = top + 1e-2;
    }

    /// Drive scale and spin from smoothed channels and move the camera;
    /// everything uses wall-clock `dt`.
    pub fn update(&mut self, channels: &[Channel], anim: &AnimParams, dt: f32) {
        let span = (anim.max_spin - anim.idle_spin).max(0.0);
        for i in 0..self.solids.len() {
            let c = channels.get(i).copied().unwrap_or_default();
            let (level, sy, sx) = if c.available { (c.level, c.spin_y, c.spin_x) } else { (0.0, 0.0, 0.0) };
            let scale = anim.min_scale + (anim.max_scale - anim.min_scale) * level.clamp(0.0, 1.0);
            self.set_scale(i, scale);
            let yaw_rate = anim.idle_spin + span * sy.clamp(0.0, 1.0);
            let pitch_rate = 0.5 * anim.idle_spin + span * sx.clamp(0.0, 1.0);
            let (p, y) = &mut self.angles[i];
            *p = (*p + pitch_rate * dt).rem_euclid(TAU);
            *y = (*y + yaw_rate * dt).rem_euclid(TAU);
        }
        self.refresh_transforms();
        self.rig.advance(dt);
        self.camera = self.rig.camera();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full(n: usize) -> Vec<Channel> {
        vec![Channel { available: true, level: 1.0, spin_y: 1.0, spin_x: 1.0, activity: 1.0 }; n]
    }

    #[test]
    fn solids_rest_above_ground_and_inside_slab() {
        let anim = AnimParams::default();
        let mut s = Scene::demo(&anim);
        s.update(&full(5), &anim, 0.1);
        for inst in &s.solids {
            assert!(inst.center.y - inst.scale >= HOVER - 1e-5);
            assert!(inst.center.y + inst.scale <= s.slab_top);
            assert!((inst.scale - anim.max_scale * BASE_RADIUS).abs() < 1e-6);
        }
        assert!((s.ring_pos[0] - Vec3A::new(0.0, 0.0, MIN_RING_RADIUS)).length() < 1e-5, "first slot front-centre");
    }

    fn min_gap(pos: &[Vec3A]) -> f32 {
        let mut m = f32::MAX;
        for i in 0..pos.len() {
            for j in i + 1..pos.len() {
                m = m.min((pos[i] - pos[j]).length());
            }
        }
        m
    }

    #[test]
    fn rings_never_overlap() {
        for n in 1..=16 {
            let (pos, ring) = ring_layout(&vec![None; n]);
            assert!(min_gap(&pos) >= SPACING - 1e-3, "n={n}");
            assert_eq!(ring.iter().filter(|&&r| r == 0).count(), n.min(RING_CAPACITY));
        }
        // single solid sits at the centre
        assert_eq!(ring_layout(&[None]).0[0], Vec3A::ZERO);
    }

    #[test]
    fn explicit_rings_are_honoured_and_compacted() {
        let req = [Some(0), Some(5), Some(5), None, Some(0)];
        let (pos, ring) = ring_layout(&req);
        // ring 5 compacts to 1; the automatic slot joins ring 0 (has room)
        assert_eq!(ring, vec![0, 1, 1, 0, 0]);
        let r0 = pos[0].length();
        let r1 = pos[1].length();
        assert!(r1 >= r0 + SPACING - 1e-4, "outer ring outside inner: {r0} {r1}");
        assert!((pos[1].length() - pos[2].length()).abs() < 1e-5);
        assert!(min_gap(&pos) >= SPACING - 1e-3);
    }

    #[test]
    fn centre_ring_with_one_solid() {
        let (pos, ring) = ring_layout(&[Some(0), Some(1), Some(1), Some(1)]);
        assert_eq!(pos[0], Vec3A::ZERO);
        assert_eq!(ring, vec![0, 1, 1, 1]);
        assert!(pos[1].length() >= SPACING - 1e-4);
    }

    #[test]
    fn unavailable_uses_min_scale_and_idle_spin() {
        let anim = AnimParams::default();
        let mut s = Scene::demo(&anim);
        let before = s.angles[4];
        let off = vec![Channel { available: false, level: 1.0, spin_y: 1.0, spin_x: 1.0, activity: 1.0 }; 5];
        s.update(&off, &anim, 1.0);
        assert!((s.solids[4].scale - anim.min_scale).abs() < 1e-6);
        let dyaw = (s.angles[4].1 - before.1).rem_euclid(TAU);
        assert!((dyaw - anim.idle_spin).abs() < 1e-5);
    }

    #[test]
    fn camera_and_light_scale_with_layout() {
        let anim = AnimParams::default();
        let small = Scene::demo(&anim);
        let slots: Vec<SlotLayout> = (0..14).map(|i| SlotLayout { solid: SolidKind::ALL[i], ring: None }).collect();
        let big = Scene::new(&slots, &anim, &CameraParams::default());
        assert!(big.extent > small.extent);
        assert!((big.camera.pos - big.centroid).length() > (small.camera.pos - small.centroid).length());
        assert!(big.light.radius > small.light.radius);
    }

    #[test]
    fn instance_sdf_respects_transform() {
        let s = Scene::demo(&AnimParams::default());
        let inst = &s.solids[1]; // cube
        assert!((inst.sdf(inst.center) + inst.poly.inradius() * inst.scale).abs() < 1e-4);
        assert!(inst.sdf(inst.center + Vec3A::new(0.0, inst.scale * 1.01, 0.0)) > 0.0);
    }
}
