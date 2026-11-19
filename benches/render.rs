//! Render benchmark: 120×40 cells, default settings (2 spp, 2 bounces).
//! Acceptance target: ≥150 fps ⇒ ≤6.67 ms per frame on a modern multi-core CPU.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use platomon::ascii::{self, Ramp};
use platomon::aspect::Viewport;
use platomon::metrics::Channel;
use platomon::render::{RenderSettings, Renderer};
use platomon::scene::sdf::SolidKind;
use platomon::scene::{AnimParams, CameraParams, Scene, SlotLayout};

fn scene_at(level: f32) -> Scene {
    let anim = AnimParams::default();
    let mut s = Scene::demo(&anim);
    let ch = [Channel { available: true, level, spin_y: level, spin_x: level * 0.3, activity: level }; 5];
    s.update(&ch, &anim, 0.37);
    s
}

/// 16 solids on two rings, one of each of the 15 kinds (+1 repeat).
fn scene_16(level: f32) -> Scene {
    let anim = AnimParams::default();
    let slots: Vec<SlotLayout> =
        (0..16).map(|i| SlotLayout { solid: SolidKind::ALL[i % SolidKind::ALL.len()], ring: None }).collect();
    let mut s = Scene::new(&slots, &anim, &CameraParams::default());
    let ch = vec![Channel { available: true, level, spin_y: level, spin_x: level * 0.3, activity: level }; 16];
    s.update(&ch, &anim, 0.37);
    s
}

fn bench(c: &mut Criterion) {
    let (w, h) = (120usize, 40usize);
    let settings = RenderSettings::default();
    let ramp = Ramp::default();

    for (name, level) in [("render_120x40_idle", 0.0), ("render_120x40_full", 1.0)] {
        let scene = scene_at(level);
        let mut r = Renderer::new();
        c.bench_function(name, |b| b.iter(|| black_box(r.render(black_box(&scene), w, h, &settings)).len()));
    }

    let scene = scene_16(1.0);
    let mut r = Renderer::new();
    c.bench_function("render_120x40_16_solids_full", |b| {
        b.iter(|| black_box(r.render(black_box(&scene), w, h, &settings)).len())
    });

    let scene = scene_at(1.0);
    let mut r = Renderer::new();
    let mut out = String::with_capacity(w * h * 2);
    let vp = Viewport { x: 0, y: 0, w, h, fit: None };
    c.bench_function("frame_120x40_full_with_ascii", |b| {
        b.iter(|| {
            let buf = r.render(&scene, w, h, &settings);
            ascii::compose_terminal(buf, &vp, (w, h), &ramp, 1.0, None, &mut out);
            black_box(out.len())
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
