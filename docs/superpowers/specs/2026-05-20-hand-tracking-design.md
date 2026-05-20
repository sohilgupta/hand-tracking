# Hand Tracking Interactive Art Experience — Design Spec
**Date:** 2026-05-20  
**Status:** Approved  

---

## Overview

A fullscreen, browser-based interactive webcam art installation. The user's hand, tracked via MediaPipe Hands, drives real-time WebGL visual effects — four gesture modes (fist, peace, pointing, open hand), each mapping to a distinct cinematic shader aesthetic. The experience is premium, restrained, and fluid.

---

## Deliverables

| File | Purpose |
|---|---|
| `index.html` | Canvas stack, MediaPipe CDN imports, loading screen markup |
| `style.css` | Fullscreen layout, `cursor:none`, overlay canvas positioning, loading screen |
| `script.js` | All logic in clearly sectioned modules |
| `server.py` | `python3 -m http.server 8080` thin wrapper |

---

## Architecture: Rendering Pipeline

```
getUserMedia() → <video> (mirrored via CSS scaleX(-1))
     ↓
MediaPipe Hands (CPU) — 21 landmarks, confidence, per-frame
     ↓
TrackingEngine — normalized 0–1 coords, velocity EMA, confidence weighting, depth estimate
     ↓
GestureEngine — raw classify → hysteresis → stableGesture → blendWeights[5]
     ↓
┌─────────────────────────────┐    ┌──────────────────────────────┐
│  WebGL Canvas (bottom)      │    │  2D Overlay Canvas (top)     │
│  video texture → FBO pass 1 │    │  position:absolute over WebGL│
│  (mode shaders)             │    │  Particle system             │
│       ↓                     │    │  Fingertip orb + neon trail  │
│  FBO pass 2 (post-process)  │    │  Gesture label chip          │
│  bloom, vignette, grain,    │    │  Mic/quality indicators      │
│  color grade, lens distort  │    │  Debug overlay (D key)       │
└─────────────────────────────┘    └──────────────────────────────┘
     ↓
Fullscreen display (devicePixelRatio-aware)
```

### Key Invariants

- MediaPipe runs on every `requestAnimationFrame` at native video rate (~30fps)
- WebGL shader mode switches = uniform updates only, never recompilation
- Blend weights always sum to 1.0; two modes can crossfade simultaneously
- All tracking coordinates stay in normalized 0–1 space; pixel conversion happens only at render time
- All animation driven by `deltaTime` — consistent speed across devices and FPS targets
- Audio (`M` key opt-in) feeds a single `u_audioLevel` uniform; mic requested only when enabled

---

## File Structure: script.js Sections

```
CONFIG            — FPS target, DPR, particle pool size, smoothing constants, effect params
WebcamSetup       — getUserMedia, video element, CSS mirror, DPR-aware canvas resize + ResizeObserver
WebGLEngine       — context creation, shader compilation, FBO setup, texture management
Shaders           — GLSL source as template literals (vertex, fragment, postprocess)
TrackingEngine    — MediaPipe Hands init, landmark → normalized coords, EMA velocity, depth, confidence
GestureEngine     — finger extension classifier, hysteresis ring buffer, blend weight interpolation
ParticleSystem    — object-pooled trail particles on 2D overlay canvas, additive blending
AudioEngine       — M key toggle, AnalyserNode, smoothed level, peak detection
QualityManager    — rolling FPS monitor, adaptive tier (LOW/MEDIUM/HIGH), Q key manual cycle
UILayer           — gesture label, mic/quality indicators, loading screen fade, cursor hide
RenderLoop        — requestAnimationFrame, deltaTime, orchestrates all engines per frame
```

---

## Tracking Engine

### Inputs
- MediaPipe `NormalizedLandmarkList` (21 points, per-hand confidence)
- When two hands detected: use the hand with higher `score` (confidence)

### Outputs (all normalized 0–1)
- `fingerPos` — index fingertip (landmark 8), smoothed via lerp
- `palmPos` — palm center (landmark 0 + average of 5, 9, 13, 17), smoothed
- `velocity` — EMA of centroid delta between frames, clamped and smoothed
- `depth` — bounding box area of hand landmarks as pseudo-depth proxy
- `confidence` — MediaPipe hand score, used to weight velocity contribution

### Smoothing
- Lerp factor `0.25` for position (tunable in CONFIG)
- Velocity EMA alpha `0.15`; clamped to `[0, 1]` normalized before shader feed
- All uniforms (`u_velocity`, `u_audioLevel`, `u_depth`) additionally smoothed before shader upload to prevent frame-noise popping

### LOST_TRACKING State
- Triggered after 500ms with no hand detected
- Smoothly decays particle emission to zero
- Fades blend weights toward idle mode over 800ms
- Resets hysteresis buffer after 1200ms
- `activation pulse` plays in reverse on loss (subtle shader ramp-out)

---

## Gesture Engine

### Finger Extension Classification
Each finger classified as extended/curled by comparing tip distance from wrist (landmark 0) vs. base knuckle distance. Thumb uses angle at CMC joint.

| Gesture | Condition |
|---|---|
| Fist | All 5 curled |
| Open Hand | All 5 extended |
| Peace Sign | Index + middle extended; ring + pinky + thumb curled |
| Pointing Finger | Only index extended (thumb optionally tucked) |
| Idle | No hand / LOST_TRACKING |

### Separation of Concerns
Three distinct concepts kept separate:
- `rawGesture` — frame-by-frame classifier output (noisy)
- `stableGesture` — hysteresis-filtered stable mode
- `blendWeights[5]` — smooth animated floats driving actual rendering

### Hysteresis
- Ring buffer of last 8 frames of `rawGesture` votes
- Mode switch triggers only when a new gesture wins ≥ 6/8 votes
- Prevents flicker at classification boundaries

### Blend Weight Interpolation
- On confirmed switch: target weight ramps 0→1 over ~400ms via `smoothstep`
- Outgoing mode ramps 1→0 simultaneously
- All weights always sum to 1.0
- During crossfade: total distortion contribution normalized to prevent over-distortion

### Effect Intensity
Gesture identity selects mode. Intensity is a separate vector:
- `velocity` (clamped, smoothed) — scales distortion, particle speed, glitch amount
- `audioLevel` (smoothed) — scales VHS aberration, ripple secondary layer
- `depth` — scales spotlight radius, ripple frequency, VHS intensity
- Velocity spikes damped via `smoothstep` easing before shader upload

---

## GLSL Shader Architecture

### Single Fragment Shader Program (one string, four sections)

**Section 1 — Uniforms**
```glsl
uniform sampler2D u_texture;       // video frame
uniform float u_time;              // total elapsed seconds
uniform float u_delta;             // frame delta time
uniform vec2  u_resolution;        // canvas size in pixels
uniform vec4  u_blendWeights;      // [fist, peace, point, open]
uniform float u_idleWeight;        // blend weight for idle mode
uniform vec2  u_fingerPos;         // normalized fingertip
uniform vec2  u_palmPos;           // normalized palm center
uniform float u_velocity;          // 0–1 smoothed
uniform float u_audioLevel;        // 0–1 smoothed (0 if mic off)
uniform float u_depth;             // 0–1 pseudo-depth
uniform float u_qualityTier;       // 0=LOW, 1=MED, 2=HIGH
uniform float u_activationT;       // 0–1 hand appear/disappear ramp
uniform float u_lensDistortion;    // global mild barrel (always-on)
```

**Section 2 — Utility Functions**
```
hash(vec2)           — fast pseudo-random
noise2D(vec2)        — value noise
fbm(vec2, octaves)   — fractal brownian motion
chromaticAberration(sampler2D, vec2, float, bool highlightOnly)
barrel(vec2, float)  — barrel/pincushion distortion
scanline(vec2, float) — horizontal scanline darkening
vignette(vec2, float) — radial falloff
rippleDisplace(vec2, vec2 origin, float time, float amp, float freq)
edgeAttenuation(vec2) — reduce displacement near screen edges
```

**Section 3 — Mode Functions**
Each returns `vec4` color for that UV coordinate.

- `modeDither(uv)` — luma quantization, Bayer dither, column-shift glitch, grain overlay. Velocity shrinks quantization levels (8→2). Velocity spikes trigger horizontal tear burst (2–3 frames).
- `modeVHS(uv)` — barrel first, then sample; R/G/B aberration at highlight edges (luminance-weighted, avoids muddy separation); horizontal scanlines; random tracking glitch strips; slow VHS noise drift; audio adds to aberration + glitch probability.
- `modeSpotlight(uv)` — pass-through sample; spotlight mask via `smoothstep` with noisy/irregular feathering for organic feel; dark surround `0.08` multiply; radius driven by `u_depth`; bloom emphasised in post-pass.
- `modeRipple(uv)` — 3-ring wave displacement from palm, phase-offset per ring; amplitude scaled by velocity; frequency scaled by depth; edge attenuation to prevent border stretch; audio adds high-freq secondary shimmer.
- `modeIdle(uv)` — pass-through; ambient grain at 0.06; soft low-amplitude scanline flicker; vignette pulses on 4s sine wave.

**Section 4 — Compositor (main)**
```glsl
void main() {
  // 1. Apply global mild barrel lens distortion to UV (always-on, very subtle)
  // 2. Sample each mode at distorted UV
  // 3. Mix all 5 modes by blend weights (sum = 1.0)
  // 4. Normalize distortion contribution during crossfades
  // 5. Apply activation ramp (u_activationT): shader intensity scales in over ~500ms
  // 6. Output to FBO (pass 1) or screen (pass 2)
}
```

### Post-Processing Pass (separate FBO → screen, postprocess fragment shader)

Applied once after all mode blending:

| Effect | Detail |
|---|---|
| **Bloom / Glow** | Threshold bright pixels → 3 box-blur passes (count scales with quality tier) → add back additively; luminance-preserving clamp to avoid washout; intensity 0.3 base, +0.4 in spotlight mode |
| **Vignette** | Radial falloff; idle: breathing via `sin(time * 0.25)`; VHS: harder edge; spotlight: suppressed |
| **Film Grain** | `hash(uv + fract(time * 0.1))` per frame; 0.04 base; +0.15 in dither mode; grain resolution scales with quality tier |
| **Color Grading** | Subtle S-curve contrast lift; slightly raised blacks; mild cool-to-warm tint; keeps all modes visually cohesive |
| **Procedural Variation** | Tiny time-varying offsets on VHS scanline drift, ripple phase, grain pattern — prevents static/repetitive feel |

---

## Particle System

- **Pool size:** 300 (HIGH), 150 (MED), 75 (LOW) — reused, never allocated per frame
- **Fingertip orb:** glowing white circle, radius ~12px, radial gradient, additive blend
- **Neon trail:** particles emitted at fingertip each frame, inherit velocity, fade by alpha and lifetime
- **Trail properties:** soft glow radius, velocity-based trail length (fast = longer), additive compositing on 2D overlay canvas
- **Smoothing:** fingertip position lerp'd before particle emission to reduce jitter
- **On LOST_TRACKING:** emission stops; existing particles finish their lifetime naturally (no abrupt cutoff)

---

## Audio Engine (M key opt-in)

- `M` key: first press requests `getUserMedia({audio:true})` — browser permission dialog
- On grant: creates `AnalyserNode`, reads `getByteFrequencyData` each frame
- `audioLevel` = RMS of frequency data, smoothed with EMA alpha 0.2
- Peak detection: spike > threshold triggers `pulse` boolean for 3 frames
- Mic indicator in UI: small animated dot in corner when active
- If denied: indicator shows "mic unavailable", no retry
- Audio level fed as `u_audioLevel` uniform; zero when mic off

---

## Quality Manager

### Quality Tiers

| Tier | Particle pool | Bloom passes | Ripple rings | Grain resolution |
|---|---|---|---|---|
| LOW | 75 | 1 | 2 | 0.5× |
| MEDIUM | 150 | 2 | 3 | 0.75× |
| HIGH | 300 | 3 | 3 | 1.0× |

### Adaptive Scaling
- Rolling FPS monitor over 60-frame window
- If FPS < target × 0.85 for 2 consecutive seconds → step down tier
- If FPS > target × 0.95 for 5 seconds → step up tier (with hysteresis to prevent thrashing)
- `Q` key cycles manually through LOW / MEDIUM / HIGH
- Tier shown in debug overlay and quality indicator chip

---

## WebGL Fallback

If `getContext('webgl')` or `getContext('webgl2')` fails:
- Show brief "WebGL unavailable — using simplified mode" toast
- Fall back to Canvas 2D: webcam displayed, particles active, gesture label shown
- No shader effects; subtle CSS filter approximations where possible
- Tracking and gesture detection unchanged

---

## DPR & Resize Handling

- Canvas physical size = `window.innerWidth/Height * devicePixelRatio`
- CSS size = `window.innerWidth/Height` (so canvas fills screen without blurring on Retina)
- `ResizeObserver` on `document.body`: rebuilds FBO textures and updates `u_resolution` uniform cleanly on resize/fullscreen change — no stretched artifacts

---

## UI Layer

| Element | Detail |
|---|---|
| Loading screen | Black `<div>` over everything; fades out once webcam stream + MediaPipe ready |
| Gesture label | Small chip bottom-center; elegant font; updates on `stableGesture` change with crossfade |
| Mic indicator | Small dot bottom-right; pulses when active; "M: mic off" hint on first load |
| Quality indicator | "Q: HIGH" small chip top-right; updates on tier change |
| Debug overlay (`D`) | Top-left panel: FPS, gesture + confidence, blend weights, particle count, quality tier |
| Cursor | `cursor: none` on `<body>`; hidden throughout |
| Activation hint | Subtle "show your hand" text fades in after 3s of idle, fades out on first detection |

---

## Idle → Active Transition

- On first hand detection after LOST_TRACKING: play activation pulse
- `u_activationT` ramps 0→1 over ~500ms using smoothstep
- Shader intensity multiplied by `u_activationT` — effects ramp in smoothly, feel intentional
- Reverse on LOST_TRACKING (ramp out over ~800ms)

---

## Coordinate System

- All tracking: normalized 0–1 (origin top-left, matching MediaPipe convention)
- Shader UV: standard `[0,1]²` with Y flipped to match WebGL convention
- Conversion to canvas pixels only in ParticleSystem at render time
- Simplifies resize and DPR handling throughout

---

## Performance Notes

- Gesture classification runs inside `requestAnimationFrame` (not a worker) — fast enough at current complexity
- MediaPipe model is loaded from CDN on first run; Loading screen covers this
- OffscreenCanvas not used now but render separation is clean enough to migrate later
- Object pooling in ParticleSystem eliminates GC pressure during fast motion
- All uniform uploads batched before `gl.drawArrays` call

---

## Local Server

```bash
python3 server.py   # starts http.server on port 8080
# then open http://localhost:8080
```

`server.py` is a 5-line wrapper — no dependencies.

---

## Effect Design Principles

- **Restraint first** — effects are subtle by default; intensity scales with movement/audio, not maxed out constantly
- **Velocity damping** — spikes clamped via smoothstep before reaching shaders; no visual explosions
- **Coherence** — unified color grading in post-pass keeps all four modes visually consistent
- **Organic feel** — noisy spotlight feathering, procedural variation in VHS drift + ripple phase, breathing vignette — nothing feels mechanical or static
