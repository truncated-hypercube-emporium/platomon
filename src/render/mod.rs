//! Per-pixel ray generation, parallel over rows, plus the adaptive-quality governor.

pub mod march;
pub mod trace;

use glam::Vec3A;
use rayon::prelude::*;

use crate::scene::Scene;
pub use trace::TraceSettings;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderSettings {
    pub spp: u32,
    pub trace: TraceSettings,
    /// Terminal cell width / height (≈0.5). Row spacing = column spacing / char_aspect.
    pub char_aspect: f32,
    /// Frame the camera for this picture ratio (width / height). The
    /// viewport shows the centre of a picture of this ratio that covers it:
    /// identical for a letterboxed viewport, cropped for a full-terminal one.
    pub fit_aspect: Option<f32>,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self { spp: 2, trace: TraceSettings::default(), char_aspect: 0.5, fit_aspect: None }
    }
}

/// Renders radiance into a reused buffer of `w * h` floats.
#[derive(Default)]
pub struct Renderer {
    buf: Vec<f32>,
    frame: u64,
}

#[inline]
fn seed(frame: u64, row: usize) -> u64 {
    (frame.wrapping_add(1)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (row as u64 + 1).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
}

impl Renderer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn render(&mut self, scene: &Scene, w: usize, h: usize, rs: &RenderSettings) -> &[f32] {
        self.frame = self.frame.wrapping_add(1);
        self.buf.resize(w * h, 0.0);
        if w == 0 || h == 0 {
            return &self.buf;
        }
        let cam = scene.camera;
        let aspect = rs.char_aspect.max(0.05);
        let mut img_w = w as f32;
        let mut img_h = h as f32 / aspect;
        if let Some(r) = rs.fit_aspect.filter(|r| r.is_finite() && *r > 0.0) {
            if img_w / img_h > r {
                img_h = img_w / r;
            } else {
                img_w = img_h * r;
            }
        }
        // Fit both minimum FOVs: whichever needs the wider view wins.
        let k = (2.0 * (cam.hfov * 0.5).tan() / img_w).max(2.0 * (cam.vfov * 0.5).tan() / img_h);
        let (half_w, half_h) = (w as f32 * 0.5, h as f32 * 0.5);
        let spp = rs.spp.max(1);
        let inv_spp = 1.0 / spp as f32;
        let frame = self.frame;
        let ts = rs.trace;

        self.buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let mut rng = fastrand::Rng::with_seed(seed(frame, y));
            for (x, out) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for _ in 0..spp {
                    let (jx, jy) = if spp == 1 { (0.5, 0.5) } else { (rng.f32(), rng.f32()) };
                    let u = (x as f32 + jx - half_w) * k;
                    let v = (half_h - y as f32 - jy) / aspect * k;
                    let rd: Vec3A = (cam.forward + cam.right * u + cam.up * v).normalize();
                    sum += trace::radiance(scene, cam.pos, rd, &mut rng, &ts);
                }
                *out = sum * inv_spp;
            }
        });
        &self.buf
    }
}

/// Steps quality down (spp → 1, then bounces → 1) when frames miss their
/// budget, and back up when there is clear headroom.
#[derive(Debug, Clone)]
pub struct QualityGovernor {
    level: u8,
    load: f32,
    over_for: f32,
    under_for: f32,
}

impl Default for QualityGovernor {
    fn default() -> Self {
        Self { level: 0, load: 0.0, over_for: 0.0, under_for: 0.0 }
    }
}

impl QualityGovernor {
    pub const MAX_LEVEL: u8 = 2;

    pub fn level(&self) -> u8 {
        self.level
    }

    /// `render_secs`: time spent rendering this frame; `dt`: frame wall time.
    pub fn update(&mut self, render_secs: f32, target_fps: f32, dt: f32) {
        let budget = 1.0 / target_fps.max(1.0);
        let a = 1.0 - (-dt / 0.3f32).exp();
        self.load += (render_secs / budget - self.load) * a;
        if self.load > 0.9 {
            self.over_for += dt;
            self.under_for = 0.0;
        } else if self.load < 0.35 {
            self.under_for += dt;
            self.over_for = 0.0;
        } else {
            self.over_for = 0.0;
            self.under_for = 0.0;
        }
        if self.over_for > 0.5 && self.level < Self::MAX_LEVEL {
            self.level += 1;
            self.over_for = 0.0;
        } else if self.under_for > 2.0 && self.level > 0 {
            self.level -= 1;
            self.under_for = 0.0;
        }
    }

    pub fn apply(&self, base: RenderSettings) -> RenderSettings {
        let mut s = base;
        if self.level >= 1 {
            s.spp = 1;
        }
        if self.level >= 2 {
            s.trace.bounces = s.trace.bounces.min(1);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{AnimParams, Scene};

    #[test]
    fn renders_requested_size_with_finite_values() {
        let scene = Scene::demo(&AnimParams::default());
        let mut r = Renderer::new();
        let buf = r.render(&scene, 40, 12, &RenderSettings::default());
        assert_eq!(buf.len(), 480);
        assert!(buf.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(buf.iter().any(|v| *v > 0.05), "something is lit");
        assert_eq!(r.render(&scene, 0, 0, &RenderSettings::default()).len(), 0);
    }

    #[test]
    fn governor_degrades_and_recovers() {
        let mut g = QualityGovernor::default();
        for _ in 0..100 {
            g.update(0.05, 60.0, 1.0 / 20.0); // 3x over budget
        }
        assert_eq!(g.level(), 2);
        let s = g.apply(RenderSettings::default());
        assert_eq!((s.spp, s.trace.bounces), (1, 1));
        for _ in 0..1000 {
            g.update(0.001, 60.0, 1.0 / 60.0);
        }
        assert_eq!(g.level(), 0);
    }
}
