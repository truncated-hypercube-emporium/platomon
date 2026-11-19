# platomon 1.2

A terminal system monitor: one path-traced polyhedron per metric, rendered in monochrome ASCII on
concentric rings. Each solid's scale and spin follow its metric; the frame rate (5–150 fps) follows
total activity.

Keys: `s` toggles the stats bar, `q` / `Esc` / `Ctrl-C` quit.

## Launch parameters

```
platomon [OPTIONS]
```

| Option | Syntax | Default |
|---|---|---|
| **General** | | |
| `--config` | `PATH` | `<config dir>/platomon/config.toml` if present |
| `--list` | `solids\|metrics\|aspects\|sensors\|patterns` | prints IDs (sensors: this machine's disks/adapters) and exits |
| `--print-config` | flag | print effective config as TOML |
| `--snapshot` | `COLSxROWS` | render one frame to stdout, no terminal needed |
| `--snapshot-time` | seconds of animation to simulate first | 0.3 |
| `--stats` | flag | stats bar visible at start |
| **Metrics & solids** | | |
| `--show` | `METRIC[,METRIC…]`, max 16 | `cpu,ram,disk,net,gpu` |
| `--solid` | `METRIC=SOLID`, repeatable | built-in mapping |
| `--ring` | `METRIC=N`, repeatable (0 = innermost) | automatic |
| **Rendering** | | |
| `--spp` | `N` (1–64) | 2 |
| `--bounces` | `N` (0–8) | 2 |
| `--fps-min` / `--fps-max` | `FPS` | 5 / 150 |
| `--ramp` | `STR`, darkest first | `" .:-=+*#%@"` |
| `--char-aspect` | cell width ÷ height | 0.5 |
| **Display ratio** | | |
| `--aspect` | `fill`, preset or `W:H` | `fill` |
| `--aspect-mode` | `letterbox\|crop` | `letterbox` |
| **Camera** | | |
| `--camera` | `static\|orbit` | `static` |
| `--orbit-period` | seconds per revolution | 60 |
| `--orbit-dir` | `random\|cw\|ccw` (seen from above) | `random` |
| `--orbit-radius` | world units | fit to layout |
| `--orbit-height` | world units | fit to layout |
| `--orbit-bob` | vertical drift amplitude, 0 = level | 0.3 |
| `--seed` | `U64`, makes `random` repeatable | random |
| **Background** | | |
| `--bg-lum` | `0..10`: 5 = original, **0 = no background surface** | 5 |
| `--bg-pattern` | `solid\|checker\|diamond\|brick\|triangles\|hex\|radial` | `solid` |
| `--bg-contrast` | `0..1`, tile tone spread around `--bg-lum` | 0.5 |
| `--bg-scale` | tile size, world units (a full-size solid is 2 across) | 1.0 |
| `--bg-scroll` | `DX:DZ` world units/s, e.g. `0.5:0` or `-0.2:0.3` | `0:0` |
| `--bg-sway` | radius of a circular back-and-forth translation; 0 = off | 0 |
| `--bg-sway-period` | seconds per sway cycle | 10 |
| `--bg-rotate` | `off\|cw\|ccw`, about the camera's focal point (seen from above) | `off` |
| `--bg-rotate-period` | seconds per revolution | 60 |

The orbit circles the centroid of all solids and advances with wall-clock time, so its speed doesn't
depend on fps.

### Background

The ground plane's mean brightness follows `--bg-lum` linearly: 5 is the original ground and 10 is
twice as bright. At 0 the plane is removed entirely, so the solids float in a black void with no
ground or contact shadows. Patterns alternate their tiles above and below that mean by
`--bg-contrast`, so changing the pattern doesn't change the overall brightness. `hex` uses three
tones, and `radial` is a polar checkerboard centred on the focal point. Far-away tiles smaller than
a character cell fade to the mean instead of flickering.

Motion is set in the pattern's own frame and runs on wall-clock time. `--bg-scroll` translates at a
constant velocity, `--bg-sway` adds a circular back-and-forth, and `--bg-rotate` spins the pattern
about the camera's focal point (the layout centroid the camera looks at). Rotation and the
`--camera orbit` direction use the same `cw`/`ccw` convention, so they can turn with or against
each other. With `--bg-pattern solid` (the default) motion has nothing visible to move.

| Pattern | Description |
|---|---|
| `solid` | uniform (original look) |
| `checker` | square checkerboard |
| `diamond` | checkerboard rotated 45° |
| `brick` | checkerboard, alternate rows offset half a tile |
| `triangles` | squares split on the diagonal, alternating |
| `hex` | hexagonal tiles, three tones |
| `radial` | rings × 12 sectors around the focal point |

### Metrics

| Metric | Scale | Spin | Default solid |
|---|---|---|---|
| `cpu` | utilisation (minus platomon's own) | utilisation | icosahedron |
| `ram` | used ÷ total | rate of change | cube |
| `disk` | all disks, read+write | Y = read, X = write | octahedron |
| `net` | all adapters, rx+tx | Y = rx, X = tx | tetrahedron |
| `gpu` | busiest GPU's utilisation | memory-controller util, else utilisation | dodecahedron |
| `nic[:NAME]` | one adapter, rx+tx | Y = rx, X = tx | rhombic-dodecahedron |
| `disk-dev[:NAME]` | one disk/volume, read+write | Y = read, X = write | truncated-cube |

`nic` and `disk-dev` without a name follow the busiest instance and can appear several times with
different names. `platomon --list sensors` shows the names this machine has. Disks also match by
mount point, e.g. `disk-dev:C:\` or `disk-dev:/home`.

Every metric here works on Linux, macOS and Windows with any GPU vendor. Metrics that some operating
systems or vendors can't provide were left out: temperatures, CPU frequency (Windows reports a fixed
base clock), and GPU clocks/PCIe throughput (NVIDIA only). When a metric is unavailable on a
particular machine, its solid sits at minimum scale with idle spin and the stats bar shows `n/a`.

### Solids

| Family | IDs |
|---|---|
| Platonic | `tetrahedron`, `cube`, `octahedron`, `dodecahedron`, `icosahedron` |
| Archimedean | `truncated-tetrahedron`, `cuboctahedron`, `truncated-cube`, `truncated-octahedron`, `rhombicuboctahedron`, `icosidodecahedron`, `truncated-icosahedron`, `truncated-dodecahedron` |
| Catalan | `rhombic-dodecahedron`, `rhombic-triacontahedron` |

### Rings

Solids are placed on concentric rings. Without `--ring`, slots fill ring 0 (up to 8), then ring 1,
and so on. Ring numbers only set the order of the rings; empty ones are skipped, so `--ring nic=5`
on its own just makes an outer ring. A ring holding a single solid puts it at the centre. Odd rings
are offset by half a step so neighbouring rings interleave.

### Aspect presets

`vga` 4:3, `svga` 4:3, `xga` 4:3, `sxga` 5:4, `wxga` 16:10, `hd` 16:9, `wxga+` 16:10, `fhd` 16:9,
`wuxga` 16:10, `qhd` 16:9, `uwfhd` 64:27, `uwqhd` 43:18, `uhd` 16:9, `square` 1:1, or any `W:H`.
`letterbox` shrinks the picture to the ratio inside the terminal; `crop` fills the terminal with a
picture framed for the ratio and cuts off the overflow.

### Examples

```
platomon --camera orbit --orbit-period 90 --aspect uwqhd
platomon --show cpu,ram,gpu,nic:Ethernet,nic:Wi-Fi --ring nic=1 --solid nic:Wi-Fi=rhombic-triacontahedron
platomon --show "cpu,disk-dev:C:\,disk-dev:D:\" --solid disk-dev=truncated-octahedron --stats
platomon --bg-pattern checker --bg-contrast 0.8 --bg-scale 1.5 --bg-scroll 0.4:0 --bg-rotate cw --bg-rotate-period 30
platomon --bg-lum 0 --camera orbit                       # solids floating in a void
```

## Config file

All options have TOML keys (dashes become underscores); see `config.example.toml`. Solid and ring
choices go in tables. A kind-wide key like `nic` applies to every `nic:*` slot, and an exact key
wins over it. Command-line options override the file.

```toml
show = ["cpu", "ram", "nic:Ethernet", "nic:Wi-Fi"]
camera = "orbit"
aspect = "wxga"

[solids]
cpu = "truncated-icosahedron"
nic = "rhombic-dodecahedron"

[rings]
nic = 1

[weights]   # activity-score weight per metric kind
cpu = 2.0
```

## Build

```
cargo build --release                       # with NVIDIA NVML support (loaded at runtime)
cargo build --release --no-default-features # without
cargo test
cargo bench --bench render
```

## Notes

- **GPU backends**, in order: NVML (feature `nvidia`, no link-time dependency) → Linux sysfs
  `gpu_busy_percent` (AMD) → sysinfo `Gpus` (Windows PDH, macOS).
- **Feedback guard**: platomon's own CPU (per-core % ÷ cores) is subtracted from system CPU.
- **Activity score**: weighted mean over available metrics. For RAM it uses the rate of change, since
  a fill level isn't activity.
- **Adaptive quality**: when render time exceeds ~90% of the frame budget, spp drops to 1, then
  bounces to 1 (`q-1` / `q-2` in the stats bar); restored once there is headroom.
- **Polyhedra** are intersections of face planes. Vertices are recovered from the planes, and tests
  check face/vertex counts, unit circumradius and equal edge lengths for every solid.
- Every frame repaints the whole terminal area, margins included, so it doesn't depend on the terminal
  honouring clear-screen codes. Terminal size is polled each frame, for hosts that don't send resize
  events.
