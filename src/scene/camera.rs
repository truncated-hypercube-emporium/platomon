//! Camera, and the rig that places it: static, or orbiting the layout's centroid.

use glam::Vec3A;
use serde::{Deserialize, Serialize};
use std::f32::consts::TAU;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CameraMode {
    #[default]
    Static,
    Orbit,
}

impl FromStr for CameraMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "static" => Ok(Self::Static),
            "orbit" => Ok(Self::Orbit),
            _ => Err(format!("unknown camera '{s}' (static|orbit)")),
        }
    }
}

/// Orbit direction, seen from above.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OrbitDir {
    #[default]
    Random,
    Cw,
    Ccw,
}

impl FromStr for OrbitDir {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "random" => Ok(Self::Random),
            "cw" => Ok(Self::Cw),
            "ccw" => Ok(Self::Ccw),
            _ => Err(format!("unknown orbit direction '{s}' (random|cw|ccw)")),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub pos: Vec3A,
    pub forward: Vec3A,
    pub right: Vec3A,
    pub up: Vec3A,
    /// Minimum horizontal / vertical field of view, radians (fit whichever is tighter).
    pub hfov: f32,
    pub vfov: f32,
}

impl Camera {
    pub fn look_at(pos: Vec3A, target: Vec3A, hfov_deg: f32, vfov_deg: f32) -> Self {
        let forward = (target - pos).normalize();
        let right = forward.cross(Vec3A::Y).normalize();
        let up = right.cross(forward);
        Self { pos, forward, right, up, hfov: hfov_deg.to_radians(), vfov: vfov_deg.to_radians() }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CameraParams {
    pub mode: CameraMode,
    pub period: f32,
    pub dir: OrbitDir,
    pub radius: Option<f32>,
    pub height: Option<f32>,
    pub bob: f32,
    pub seed: Option<u64>,
}

impl Default for CameraParams {
    fn default() -> Self {
        Self { mode: CameraMode::Static, period: 60.0, dir: OrbitDir::Random, radius: None, height: None, bob: 0.3, seed: None }
    }
}

/// Distance and height per unit of layout extent (tuned on the 5-solid ring).
const RADIUS_PER_EXTENT: f32 = 2.32;
const HEIGHT_PER_EXTENT: f32 = 1.4;
const TARGET_Y: f32 = 0.55;
const HFOV_DEG: f32 = 50.0;
const VFOV_DEG: f32 = 30.0;

#[derive(Debug, Clone)]
pub struct CameraRig {
    mode: CameraMode,
    /// +1 = counter-clockwise from above, −1 = clockwise.
    dir: f32,
    omega: f32,
    radius: f32,
    height: f32,
    bob: f32,
    angle: f32,
    centroid: Vec3A,
}

impl CameraRig {
    /// `centroid`: mean solid position; `extent`: layout radius around it.
    pub fn new(p: &CameraParams, centroid: Vec3A, extent: f32) -> Self {
        let dir = match p.dir {
            OrbitDir::Ccw => 1.0,
            OrbitDir::Cw => -1.0,
            OrbitDir::Random => {
                let mut rng = p.seed.map_or_else(fastrand::Rng::new, fastrand::Rng::with_seed);
                if rng.bool() { 1.0 } else { -1.0 }
            }
        };
        Self {
            mode: p.mode,
            dir,
            omega: TAU / p.period.max(0.1),
            radius: p.radius.unwrap_or(RADIUS_PER_EXTENT * extent),
            height: p.height.unwrap_or(HEIGHT_PER_EXTENT * extent),
            bob: p.bob,
            angle: 0.0,
            centroid,
        }
    }

    pub fn direction(&self) -> f32 {
        self.dir
    }

    pub fn angle(&self) -> f32 {
        self.angle
    }

    /// Advance by wall-clock `dt` (orbit only).
    pub fn advance(&mut self, dt: f32) {
        if self.mode == CameraMode::Orbit {
            self.angle = (self.angle + self.dir * self.omega * dt).rem_euclid(TAU);
        }
    }

    pub fn camera(&self) -> Camera {
        let (s, c) = self.angle.sin_cos();
        let bob = if self.mode == CameraMode::Orbit { self.bob * (2.0 * self.angle).sin() } else { 0.0 };
        let pos = self.centroid + Vec3A::new(self.radius * s, self.height + bob, self.radius * c);
        let target = self.centroid + Vec3A::new(0.0, TARGET_Y, 0.0);
        Camera::look_at(pos, target, HFOV_DEG, VFOV_DEG)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(mode: CameraMode, dir: OrbitDir) -> CameraParams {
        CameraParams { mode, dir, period: 10.0, ..CameraParams::default() }
    }

    #[test]
    fn static_never_moves() {
        let mut r = CameraRig::new(&params(CameraMode::Static, OrbitDir::Ccw), Vec3A::ZERO, 3.7);
        let before = r.camera().pos;
        r.advance(5.0);
        assert_eq!(r.camera().pos, before);
        assert!(before.z > 0.0 && before.x.abs() < 1e-6, "in front, on +z");
    }

    #[test]
    fn orbit_circles_centroid_at_period() {
        let c = Vec3A::new(1.0, 0.0, -2.0);
        let mut r = CameraRig::new(&CameraParams { bob: 0.0, ..params(CameraMode::Orbit, OrbitDir::Ccw) }, c, 3.7);
        let d0 = (r.camera().pos - c) * Vec3A::new(1.0, 0.0, 1.0);
        r.advance(2.5); // quarter turn
        let p = r.camera().pos - c;
        assert!(((p * Vec3A::new(1.0, 0.0, 1.0)).length() - d0.length()).abs() < 1e-4, "constant radius");
        assert!(p.x > 0.0 && p.z.abs() < 1e-3, "ccw from +z goes to +x: {p:?}");
        r.advance(7.5);
        assert!(r.angle() < 1e-4 || (TAU - r.angle()) < 1e-4, "full turn after one period");
        // looks at the centroid
        let cam = r.camera();
        let to = (c + Vec3A::new(0.0, TARGET_Y, 0.0) - cam.pos).normalize();
        assert!(cam.forward.dot(to) > 0.9999);
    }

    #[test]
    fn directions() {
        let cw = CameraRig::new(&params(CameraMode::Orbit, OrbitDir::Cw), Vec3A::ZERO, 1.0);
        assert_eq!(cw.direction(), -1.0);
        let a = CameraRig::new(&CameraParams { seed: Some(42), ..params(CameraMode::Orbit, OrbitDir::Random) }, Vec3A::ZERO, 1.0);
        let b = CameraRig::new(&CameraParams { seed: Some(42), ..params(CameraMode::Orbit, OrbitDir::Random) }, Vec3A::ZERO, 1.0);
        assert_eq!(a.direction(), b.direction(), "seeded random is repeatable");
        let seen: std::collections::HashSet<i32> = (0..64)
            .map(|s| CameraRig::new(&CameraParams { seed: Some(s), ..params(CameraMode::Orbit, OrbitDir::Random) }, Vec3A::ZERO, 1.0).direction() as i32)
            .collect();
        assert_eq!(seen.len(), 2, "random yields both directions");
    }

    #[test]
    fn parse_enums() {
        assert_eq!("ORBIT".parse::<CameraMode>().unwrap(), CameraMode::Orbit);
        assert_eq!("cw".parse::<OrbitDir>().unwrap(), OrbitDir::Cw);
        assert!("sideways".parse::<OrbitDir>().is_err());
    }
}
