use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use crossterm::{cursor, event, execute, terminal};

use platomon::ascii::{self, Ramp, StatsLine};
use platomon::aspect::{self, AspectMode, PRESETS};
use platomon::clock::{self, FrameClock, SPIN_MARGIN};
use platomon::collector::{Collector, gpu::GpuSampler, system::SystemSampler};
use platomon::config::{Config, SlotSpec, split_assignment};
use platomon::metrics::{Channel, MetricKind, MetricsState, activity_score};
use platomon::render::{QualityGovernor, RenderSettings, Renderer, TraceSettings};
use platomon::scene::camera::{CameraMode, CameraParams, OrbitDir};
use platomon::scene::sdf::{Polyhedron, SolidKind};
use platomon::scene::{AnimParams, Scene, SlotLayout};
use platomon::ui::{self, Action};

/// Longest sleep between input polls while waiting for a frame deadline.
const INPUT_SLICE: Duration = Duration::from_millis(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ListWhat {
    Solids,
    Metrics,
    Aspects,
    Sensors,
}

/// Platonic-solid ASCII system monitor.
///
/// One path-traced solid per metric, on concentric rings. Keys: s = stats bar,
/// q / Esc / Ctrl-C = quit.
#[derive(Parser, Debug)]
#[command(name = "platomon", version, about, long_about = None)]
struct Cli {
    /// Config file (TOML). Default: <config dir>/platomon/config.toml if present.
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,
    /// Print solids, metrics, aspect presets or this machine's disks/adapters, then exit.
    #[arg(long, value_name = "WHAT")]
    list: Option<ListWhat>,
    /// Print the effective configuration as TOML and exit.
    #[arg(long)]
    print_config: bool,
    /// Render one frame to stdout (e.g. 120x40) and exit.
    #[arg(long, value_name = "COLSxROWS")]
    snapshot: Option<String>,
    /// Show the stats bar at start.
    #[arg(long)]
    stats: bool,

    // ---- metrics & solids
    /// Metrics to show, one solid each (max 16): cpu,ram,disk,net,gpu,nic[:NAME],disk-dev[:NAME].
    #[arg(long, value_delimiter = ',', value_name = "METRIC[,METRIC...]")]
    show: Option<Vec<String>>,
    /// Solid for a metric (repeatable). `nic=...` applies to every nic slot.
    #[arg(long, value_name = "METRIC=SOLID")]
    solid: Vec<String>,
    /// Ring for a metric, 0 = innermost (repeatable).
    #[arg(long, value_name = "METRIC=N")]
    ring: Vec<String>,

    // ---- rendering
    /// Samples per pixel.
    #[arg(long, value_name = "N")]
    spp: Option<u32>,
    /// Indirect bounces.
    #[arg(long, value_name = "N")]
    bounces: Option<u32>,
    /// Frame-rate floor (idle).
    #[arg(long, value_name = "FPS")]
    fps_min: Option<f32>,
    /// Frame-rate ceiling (full load).
    #[arg(long, value_name = "FPS")]
    fps_max: Option<f32>,
    /// Glyph ramp, darkest first.
    #[arg(long, value_name = "STR")]
    ramp: Option<String>,
    /// Terminal cell width / height.
    #[arg(long, value_name = "FLOAT")]
    char_aspect: Option<f32>,

    // ---- display ratio
    /// Picture ratio: fill, a preset (vga, wxga, fhd, uwqhd, ... see --list aspects) or W:H.
    #[arg(long, value_name = "PRESET|W:H")]
    aspect: Option<String>,
    /// letterbox (blank margins) or crop (fill the terminal, cut overflow).
    #[arg(long, value_name = "MODE")]
    aspect_mode: Option<AspectMode>,

    // ---- camera
    /// static or orbit (circle the centroid of the solids).
    #[arg(long, value_name = "MODE")]
    camera: Option<CameraMode>,
    /// Seconds per orbit.
    #[arg(long, value_name = "SECS")]
    orbit_period: Option<f32>,
    /// random, cw or ccw (seen from above).
    #[arg(long, value_name = "DIR")]
    orbit_dir: Option<OrbitDir>,
    /// Orbit radius in world units (default: fit the layout).
    #[arg(long, value_name = "FLOAT")]
    orbit_radius: Option<f32>,
    /// Camera height (default: fit the layout).
    #[arg(long, value_name = "FLOAT")]
    orbit_height: Option<f32>,
    /// Vertical drift while orbiting; 0 = level circle.
    #[arg(long, value_name = "FLOAT")]
    orbit_bob: Option<f32>,
    /// Seed for random choices (orbit direction).
    #[arg(long, value_name = "U64")]
    seed: Option<u64>,
}

fn fail(msg: impl std::fmt::Display) -> ! {
    eprintln!("platomon: {msg}");
    std::process::exit(2);
}

fn apply_cli(cfg: &mut Config, cli: &Cli) -> Result<(), String> {
    macro_rules! set {
        ($($f:ident),*) => { $( if let Some(v) = cli.$f.clone() { cfg.$f = v; } )* };
    }
    set!(spp, bounces, fps_min, fps_max, ramp, char_aspect, aspect, aspect_mode, camera, orbit_period, orbit_dir, orbit_bob);
    if cli.orbit_radius.is_some() {
        cfg.orbit_radius = cli.orbit_radius;
    }
    if cli.orbit_height.is_some() {
        cfg.orbit_height = cli.orbit_height;
    }
    if cli.seed.is_some() {
        cfg.seed = cli.seed;
    }
    if let Some(show) = &cli.show {
        cfg.show = show.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    }
    for a in &cli.solid {
        let (k, v) = split_assignment(a).map_err(|e| format!("--solid: {e}"))?;
        cfg.solids.insert(k, v);
    }
    for a in &cli.ring {
        let (k, v) = split_assignment(a).map_err(|e| format!("--ring: {e}"))?;
        let n: u32 = v.parse().map_err(|_| format!("--ring {a}: ring must be a number 0..=15"))?;
        if n > 15 {
            return Err(format!("--ring {a}: ring must be 0..=15"));
        }
        cfg.rings.insert(k, n);
    }
    if cli.stats {
        cfg.show_stats = true;
    }
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    let mut cfg = Config::load(cli.config.as_deref()).unwrap_or_else(|e| fail(e));
    apply_cli(&mut cfg, &cli).unwrap_or_else(|e| fail(e));
    cfg.sanitize();

    if let Some(what) = cli.list {
        list(what);
        return;
    }
    if cli.print_config {
        print!("{}", cfg.to_toml());
        return;
    }
    let slots = cfg.resolve_slots().unwrap_or_else(|e| fail(e));
    let ratio = aspect::parse_aspect(&cfg.aspect).unwrap_or_else(|e| fail(e));

    if let Some(spec) = &cli.snapshot {
        let Some((w, h)) = spec.split_once(['x', 'X']).and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
        else {
            fail("--snapshot expects COLSxROWS, e.g. 120x40");
        };
        println!("{}", snapshot(&cfg, &slots, ratio, w, h));
        return;
    }

    if let Err(e) = run(&cfg, &slots, ratio) {
        restore_terminal();
        eprintln!("platomon: {e}");
        std::process::exit(1);
    }
}

// ------------------------------------------------------------------ --list

fn list(what: ListWhat) {
    match what {
        ListWhat::Solids => {
            println!("{:<26} {:>5} {:>8}  family", "SOLID", "FACES", "VERTICES");
            for (i, k) in SolidKind::ALL.iter().enumerate() {
                let p = Polyhedron::new(*k);
                let family = match i {
                    0..=4 => "Platonic",
                    13 | 14 => "Catalan",
                    _ => "Archimedean",
                };
                println!("{:<26} {:>5} {:>8}  {family}", k.id(), p.face_planes().len(), p.vertices().len());
            }
        }
        ListWhat::Metrics => {
            println!("{:<18} {:<24} {:<42} SPIN", "METRIC", "DEFAULT SOLID", "SCALE");
            for k in MetricKind::ALL {
                let (scale, spin) = k.describe();
                let id = if k.has_instance() { format!("{}[:NAME]", k.id()) } else { k.id().to_string() };
                println!("{:<18} {:<24} {:<42} {spin}", id, k.default_solid().id(), scale);
            }
            println!("\nNAME: see --list sensors. Without NAME the busiest instance is used.");
        }
        ListWhat::Aspects => {
            println!("{:<8} {:>11}  RATIO", "PRESET", "RESOLUTION");
            for (id, w, h) in PRESETS {
                let res = if id == "square" { "-".to_string() } else { format!("{w}x{h}") };
                println!("{id:<8} {res:>11}  {:.3}", w as f32 / h as f32);
            }
            println!("\nAlso: fill (default) or any W:H, e.g. --aspect 21:9");
        }
        ListWhat::Sensors => {
            let mut sys = SystemSampler::new();
            sys.sample(0.25); // prime counters
            std::thread::sleep(Duration::from_millis(1000));
            let m = sys.sample(1.0);
            let mut gpu = GpuSampler::new();
            let g = gpu.sample();
            println!("GPU backend: {} ({})", gpu.backend_name(), g.map_or("n/a".into(), |g| format!("{:.0}%", g.util)));
            println!("CPU: {}", m.cpu_total.map_or("n/a".into(), |c| format!("{c:.0}%")));
            println!("\nDisks (use with disk-dev:NAME):");
            if m.disks.is_empty() {
                println!("  none with I/O counters (disk-dev will show n/a)");
            }
            for d in &m.disks {
                let alt = d.alt.as_deref().map(|a| format!("  [{a}]")).unwrap_or_default();
                println!("  {:<24}{alt:<14} r{:>8.2} w{:>8.2} MB/s", d.name, d.a_bps / 1e6, d.b_bps / 1e6);
            }
            println!("\nNetwork adapters (use with nic:NAME):");
            if m.nics.is_empty() {
                println!("  none");
            }
            for n in &m.nics {
                println!("  {:<38} ↓{:>8.2} ↑{:>8.2} MB/s", n.name, n.a_bps / 1e6, n.b_bps / 1e6);
            }
        }
    }
}

// ------------------------------------------------------------------ setup helpers

fn anim_params(cfg: &Config) -> AnimParams {
    AnimParams { min_scale: cfg.min_scale, max_scale: cfg.max_scale, idle_spin: cfg.idle_spin, max_spin: cfg.max_spin }
}

fn camera_params(cfg: &Config) -> CameraParams {
    CameraParams {
        mode: cfg.camera,
        period: cfg.orbit_period,
        dir: cfg.orbit_dir,
        radius: cfg.orbit_radius,
        height: cfg.orbit_height,
        bob: cfg.orbit_bob,
        seed: cfg.seed,
    }
}

fn render_settings(cfg: &Config, ratio: Option<f32>) -> RenderSettings {
    RenderSettings {
        spp: cfg.spp,
        trace: TraceSettings { bounces: cfg.bounces, max_steps: cfg.max_steps, max_dist: cfg.max_dist },
        char_aspect: cfg.char_aspect,
        fit_aspect: ratio,
    }
}

fn layouts(slots: &[SlotSpec]) -> Vec<SlotLayout> {
    slots.iter().map(|s| SlotLayout { solid: s.solid, ring: s.ring }).collect()
}

/// One frame with every slot at a fixed demo level.
fn snapshot(cfg: &Config, slots: &[SlotSpec], ratio: Option<f32>, w: usize, h: usize) -> String {
    let anim = anim_params(cfg);
    let mut scene = Scene::new(&layouts(slots), &anim, &camera_params(cfg));
    let levels = [0.7, 0.5, 0.3, 0.9, 0.0, 0.6, 0.4, 0.8];
    let ch: Vec<Channel> = (0..slots.len())
        .map(|i| {
            let l = levels[i % levels.len()];
            Channel { available: true, level: l, spin_y: l, spin_x: 0.2, activity: l }
        })
        .collect();
    scene.update(&ch, &anim, 0.3);
    let vp = aspect::viewport(w, h, ratio, cfg.aspect_mode, cfg.char_aspect);
    let mut r = Renderer::new();
    let buf = r.render(&scene, vp.w, vp.h, &render_settings(cfg, ratio));
    let mut s = String::new();
    ascii::compose_plain(buf, vp.w, vp.h, &Ramp::new(&cfg.ramp), cfg.exposure, &mut s);
    s
}

// ------------------------------------------------------------------ terminal

fn restore_terminal() {
    let mut out = io::stdout();
    let _ = execute!(out, terminal::EnableLineWrap, cursor::Show, terminal::LeaveAlternateScreen);
    let _ = terminal::disable_raw_mode();
}

struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        let guard = Self; // restores even if the next line fails
        execute!(
            io::stdout(),
            terminal::EnterAlternateScreen,
            cursor::Hide,
            terminal::DisableLineWrap,
            terminal::Clear(terminal::ClearType::All)
        )?;
        Ok(guard)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_terminal();
    }
}

/// Restore the terminal before the panic message is printed. The collector
/// thread catches its own panics, so it is left alone; anything else
/// (main, rayon workers) is fatal.
fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if std::thread::current().name() == Some("platomon-collector") {
            return;
        }
        restore_terminal();
        default(info);
    }));
}

// ------------------------------------------------------------------ main loop

fn run(cfg: &Config, slots: &[SlotSpec], ratio: Option<f32>) -> io::Result<()> {
    install_panic_hook();
    let collector = Collector::spawn(cfg.poll_hz);
    let _guard = TerminalGuard::enter()?;
    let mut out = io::stdout().lock();

    let (mut cols, mut rows) = terminal::size()?;
    let mut show_stats = cfg.show_stats;
    let mut need_clear = true;

    let anim = anim_params(cfg);
    let base = render_settings(cfg, ratio);
    let ramp = Ramp::new(&cfg.ramp);
    let specs: Vec<_> = slots.iter().map(|s| s.metric.clone()).collect();

    let mut scene = Scene::new(&layouts(slots), &anim, &camera_params(cfg));
    let mut metrics = MetricsState::new(cfg, &specs);
    let weights = metrics.weights(&cfg.weights);
    let mut fps = FrameClock::new(cfg.fps_min, cfg.fps_max, cfg.fps_curve, cfg.fps_slew);
    let mut renderer = Renderer::new();
    let mut governor = QualityGovernor::default();
    let mut frame = String::with_capacity(cols as usize * rows as usize * 2 + 1024);

    loop {
        // Wait for the frame deadline, staying responsive to input. Input is
        // polled without blocking: a blocking `event::poll(timeout)` rounds up
        // to the OS timer tick (15.6 ms on Windows), capping fps at ~64, while
        // `thread::sleep` uses a high-resolution timer.
        let deadline = fps.next_deadline();
        loop {
            while event::poll(Duration::ZERO)? {
                match ui::map_event(&event::read()?) {
                    Action::Quit => return Ok(()),
                    Action::ToggleStats => {
                        show_stats = !show_stats;
                        need_clear = true;
                    }
                    Action::Resize(w, h) => {
                        (cols, rows) = (w, h);
                        need_clear = true;
                    }
                    Action::None => {}
                }
            }
            let now = Instant::now();
            if now + SPIN_MARGIN >= deadline {
                break;
            }
            std::thread::sleep((deadline - SPIN_MARGIN - now).min(INPUT_SLICE));
        }
        clock::spin_until(deadline);

        // Resize events are not delivered by every terminal (e.g. some ConPTY
        // hosts), so also poll the size; a change triggers a full clear.
        if let Ok(size) = terminal::size()
            && size != (cols, rows)
        {
            (cols, rows) = size;
            need_clear = true;
        }

        let dt = fps.begin_frame();
        if let Some(raw) = collector.latest() {
            metrics.ingest(&raw);
        }
        metrics.tick(dt);
        let channels = metrics.channels();
        fps.update_target(activity_score(&channels, &weights), dt);
        scene.update(&channels, &anim, dt);

        let area_h = (rows as usize).saturating_sub(show_stats as usize);
        let vp = aspect::viewport(cols as usize, area_h, ratio, cfg.aspect_mode, cfg.char_aspect);
        let settings = if cfg.adaptive_quality { governor.apply(base) } else { base };

        let t0 = Instant::now();
        let buf = renderer.render(&scene, vp.w, vp.h, &settings);
        let stats_text = show_stats.then(|| {
            let extra = if governor.level() > 0 { format!(" | q-{}", governor.level()) } else { String::new() };
            ui::stats_line_fit(&metrics.readings(), fps.achieved_fps(), fps.target_fps(), &extra, cols as usize)
        });
        let stats = stats_text.as_deref().map(|text| StatsLine { text, row: area_h, width: cols as usize });
        ascii::compose_terminal(buf, &vp, (cols as usize, area_h), &ramp, cfg.exposure, stats, &mut frame);
        let render_secs = t0.elapsed().as_secs_f32();

        if need_clear {
            // Belt and braces: the frame already overwrites every cell.
            frame.insert_str(0, "\x1b[2J");
            need_clear = false;
        }
        out.write_all(frame.as_bytes())?;
        out.flush()?;

        governor.update(render_secs, fps.target_fps(), dt);
    }
}
