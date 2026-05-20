/* ═══════════════════════════════════════════════════════════════════════════
   Hand Tracking Art — script.js
   Full pipeline: webcam → MediaPipe → gesture engine → WebGL shaders →
   particle system → audio → quality manager → render loop.

   Sections (in order):
     1. CONFIG
     2. WebcamSetup
     3. WebGLEngine  (context, shaders, FBOs, uniforms)
     4. Shaders      (GLSL source literals)
     5. TrackingEngine (MediaPipe init, landmark processing, EMA smoothing)
     6. GestureEngine  (classifier, hysteresis, blend weight interpolation)
     7. ParticleSystem (object-pooled 2D overlay trail)
     8. AudioEngine    (M-key opt-in, AnalyserNode, RMS, peak detect)
     9. QualityManager (rolling FPS, adaptive tier, Q-key manual)
    10. UILayer        (gesture label, mic/quality chips, debug overlay, hints)
    11. RenderLoop     (requestAnimationFrame, deltaTime, orchestration)
   ═══════════════════════════════════════════════════════════════════════════ */

'use strict';

/* Silent debug logger — console only, no UI overlay */
const DebugLog = {
  log(msg)        { console.log(`[Init] ${msg}`); },
  error(msg, err) { console.error(`[Init] ${msg}`, err); },
};

/* ════════════════════════════════════════════════════════════════════════════
   1. CONFIG
   All tunable constants in one place. Change values here rather than hunting
   through the rest of the code.
   ════════════════════════════════════════════════════════════════════════════ */
const CONFIG = {
  // Rendering
  TARGET_FPS: 60,
  POSITION_LERP: 0.22,         // fingertip / palm smoothing factor (lower = smoother)
  VELOCITY_EMA: 0.15,          // velocity exponential moving average alpha
  UNIFORM_SMOOTH: 0.12,        // additional smoothing before uniform upload

  // Gesture hysteresis
  HYSTERESIS_BUFFER: 8,        // ring buffer size for rawGesture votes
  HYSTERESIS_THRESHOLD: 6,     // votes required to switch stableGesture

  // Blend weight animation
  BLEND_RAMP_MS: 400,          // time (ms) for mode blend 0→1 or 1→0

  // Activation ramp
  ACTIVATION_RAMP_IN_MS: 500,  // hand first appears
  ACTIVATION_RAMP_OUT_MS: 800, // hand lost
  LOST_TRACKING_MS: 500,       // delay before declaring LOST_TRACKING

  // Particle system
  PARTICLE_POOL: { HIGH: 300, MEDIUM: 150, LOW: 75 },
  PARTICLE_LIFETIME_MS: 600,   // base lifetime
  ORB_RADIUS: 12,              // fingertip orb radius in px (before DPR)

  // Audio
  AUDIO_EMA: 0.2,              // smoothing for audioLevel
  AUDIO_PEAK_THRESHOLD: 0.7,   // RMS above this = pulse
  AUDIO_PEAK_FRAMES: 3,        // pulse duration in frames

  // Quality adaptive
  FPS_WINDOW: 60,              // frames in rolling FPS average
  FPS_DOWN_THRESHOLD: 0.85,    // fraction of target FPS to trigger step-down
  FPS_DOWN_DURATION_S: 2,      // consecutive seconds below threshold
  FPS_UP_THRESHOLD: 0.95,      // fraction to allow step-up
  FPS_UP_DURATION_S: 5,        // consecutive seconds above threshold

  // Post-processing
  BLOOM_INTENSITY: 0.3,        // base bloom additive strength
  VIGNETTE_STRENGTH: 0.55,
  GRAIN_STRENGTH: 0.04,
  LENS_DISTORTION: 0.012,      // always-on mild barrel

  // UI
  ACTIVATION_HINT_DELAY_S: 3,  // seconds before "show your hand" appears

  // Skeleton rendering
  SKELETON_BONE_WIDTH: 2,       // px line width for bones
  SKELETON_JOINT_SIZE: 4,       // px radius for joint dots
  SKELETON_OPACITY: 1.0,        // 0–1 overall skeleton opacity
  SKELETON_VISIBLE: true,       // master toggle
  SKELETON_FINGER_COLORS: true, // per-finger color coding vs uniform
  SKELETON_GLOW_BLUR: 8,        // glow blur radius in px

  // Two-hand mode
  TWO_HAND_MODE: true,

  // Settings
  MIN_DETECTION_CONFIDENCE: 0.65,
};

/* Gesture indices — shared across all modules */
const G = { IDLE: 0, FIST: 1, PEACE: 2, POINT: 3, OPEN: 4 };
const GESTURE_NAMES = ['—', 'FIST', 'PEACE', 'POINT', 'OPEN HAND'];

/* Quality tier enum */
const TIER = { LOW: 0, MEDIUM: 1, HIGH: 2 };
const TIER_NAMES = ['LOW', 'MED', 'HIGH'];

/* ── MediaPipe hand landmark connections (bone pairs) ───────────────────── */
const HAND_CONNECTIONS = [
  // Thumb
  [0,1],[1,2],[2,3],[3,4],
  // Index
  [0,5],[5,6],[6,7],[7,8],
  // Middle
  [0,9],[9,10],[10,11],[11,12],
  // Ring
  [0,13],[13,14],[14,15],[15,16],
  // Pinky
  [0,17],[17,18],[18,19],[19,20],
  // Palm base arc
  [5,9],[9,13],[13,17],
];

/* Per-finger color palettes [R, G, B] — indexed by finger group */
const FINGER_COLORS = {
  thumb:  [255, 80,  80 ],  // red
  index:  [0,   255, 255],  // cyan
  middle: [80,  255, 120],  // green
  ring:   [255, 220, 80 ],  // yellow
  pinky:  [220, 80,  255],  // magenta
  palm:   [0,   255, 200],  // teal (wrist, palm arcs)
};

/* Map landmark index → finger group name */
function getLandmarkFinger(idx) {
  if (idx <= 4)  return 'thumb';
  if (idx <= 8)  return 'index';
  if (idx <= 12) return 'middle';
  if (idx <= 16) return 'ring';
  if (idx <= 20) return 'pinky';
  return 'palm';
}

/* Get color for a bone connection pair */
function getBoneColor(a, b) {
  // Use the "higher" landmark (further from wrist) to determine color
  const maxIdx = Math.max(a, b);
  return FINGER_COLORS[getLandmarkFinger(maxIdx)];
}

/* ════════════════════════════════════════════════════════════════════════════
   2. WEBCAM SETUP
   Opens getUserMedia directly (no MediaPipe Camera wrapper).
   Returns a ready video element with the live webcam stream.
   ════════════════════════════════════════════════════════════════════════════ */
const WebcamSetup = (() => {
  let videoEl = null;
  let ready   = false;

  async function init() {
    DebugLog.log('WebcamSetup: looking for <video> element...');
    videoEl = document.getElementById('webcam-video');
    if (!videoEl) {
      DebugLog.error('WebcamSetup', 'Could not find #webcam-video element!');
      throw new Error('Missing #webcam-video element');
    }
    DebugLog.log('WebcamSetup: <video> element found');

    // Check if getUserMedia is available
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      DebugLog.error('WebcamSetup', 'getUserMedia not available — are you on HTTPS or localhost?');
      throw new Error('getUserMedia not available');
    }
    DebugLog.log('WebcamSetup: getUserMedia API available');

    // This is the call that triggers the browser permission prompt
    DebugLog.log('WebcamSetup: calling getUserMedia NOW — browser should show permission prompt...');
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width:  { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 },
        },
        audio: false,
      });
    } catch (err) {
      DebugLog.error('WebcamSetup: getUserMedia FAILED', err);
      throw err;
    }

    DebugLog.log('WebcamSetup: camera stream obtained — ' + stream.getVideoTracks()[0].label);
    videoEl.srcObject = stream;

    DebugLog.log('WebcamSetup: calling video.play()...');
    try {
      await videoEl.play();
    } catch (err) {
      DebugLog.error('WebcamSetup: video.play() failed', err);
      throw err;
    }
    DebugLog.log('WebcamSetup: video playing');

    // Wait for actual video data to arrive
    DebugLog.log('WebcamSetup: waiting for video data (readyState=' + videoEl.readyState + ')...');
    await new Promise((resolve) => {
      if (videoEl.readyState >= 2) { resolve(); return; }
      videoEl.addEventListener('loadeddata', () => resolve(), { once: true });
    });

    ready = true;
    DebugLog.log('WebcamSetup: ✅ video ready ' + videoEl.videoWidth + 'x' + videoEl.videoHeight);
    return videoEl;
  }

  return { init, get videoEl() { return videoEl; }, get ready() { return ready; } };
})();

/* ════════════════════════════════════════════════════════════════════════════
   4. SHADERS (declared before WebGLEngine so they are available during init)
   GLSL source strings. All effects live here.
   ════════════════════════════════════════════════════════════════════════════ */
const Shaders = (() => {

  /* ── Shared vertex shader (used by both passes) ─────────────────────── */
  /* NOTE: We use two separate vertex shaders — the main pass flips Y to
     correct for video texture orientation, while the post pass uses
     standard UVs since it reads from FBO (already correctly oriented). */
  const VERT_MAIN = `
    attribute vec2 a_position;
    varying   vec2 v_uv;
    void main() {
      // Flip X for selfie mirror view, flip Y for video-to-GL coordinate fix
      v_uv        = vec2(1.0 - (a_position.x * 0.5 + 0.5), 1.0 - (a_position.y * 0.5 + 0.5));
      gl_Position = vec4(a_position, 0.0, 1.0);
    }
  `;

  const VERT_POST = `
    attribute vec2 a_position;
    varying   vec2 v_uv;
    void main() {
      v_uv        = a_position * 0.5 + 0.5;
      gl_Position = vec4(a_position, 0.0, 1.0);
    }
  `;

  /* ── Main fragment shader (pass 1) ──────────────────────────────────── */
  const FRAG_MAIN = `
    precision highp float;
    varying vec2 v_uv;

    /* ── Uniforms ─────────────────────────────────────────────────────── */
    uniform sampler2D u_texture;      // video frame (mirrored in JS coords)
    uniform float     u_time;         // total elapsed seconds
    uniform float     u_delta;        // frame delta (seconds)
    uniform vec2      u_resolution;   // canvas size in physical pixels
    uniform vec4      u_blendWeights; // [fist, peace, point, open]
    uniform float     u_idleWeight;   // blend weight for idle pass
    uniform vec2      u_fingerPos;    // normalized fingertip (0–1, origin TL)
    uniform vec2      u_palmPos;      // normalized palm center
    uniform float     u_velocity;     // 0–1 smoothed movement speed
    uniform float     u_audioLevel;   // 0–1 smoothed mic RMS (0 if off)
    uniform float     u_depth;        // 0–1 pseudo-depth (bbox area proxy)
    uniform float     u_qualityTier;  // 0=LOW, 1=MED, 2=HIGH
    uniform float     u_activationT;  // 0–1 hand appear/disappear ramp
    uniform float     u_lensDistort;  // always-on mild barrel amount

    /* ─────────────────────────────────────────────────────────────────────
       UTILITY FUNCTIONS
       ───────────────────────────────────────────────────────────────────── */

    /* Fast hash — pseudo-random from 2D seed */
    float hash(vec2 p) {
      p = fract(p * vec2(127.619, 311.712));
      p += dot(p, p + 19.31);
      return fract(p.x * p.y);
    }

    /* Value noise (bilinear-interpolated hash) */
    float noise2D(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      vec2 u = f * f * (3.0 - 2.0 * f); // smoothstep
      return mix(
        mix(hash(i),             hash(i + vec2(1,0)), u.x),
        mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), u.x),
        u.y
      );
    }

    /* Fractal Brownian Motion — 2 octaves (hardcoded for GLSL ES 1.00 compat) */
    float fbm(vec2 p) {
      float v = 0.0;
      v += noise2D(p)        * 0.5;
      v += noise2D(p * 2.0)  * 0.25;
      return v;
    }

    /* Barrel / pincushion distortion — positive k = barrel */
    vec2 barrel(vec2 uv, float k) {
      vec2 c = uv - 0.5;
      float r2 = dot(c, c);
      return 0.5 + c * (1.0 + k * r2);
    }

    /* Chromatic aberration — split R/G/B channels along vector */
    vec4 chromaticAberration(vec2 uv, float amt, bool highlightOnly) {
      vec4 col;
      col.r = texture2D(u_texture, uv + vec2( amt, 0.0)).r;
      col.g = texture2D(u_texture, uv                  ).g;
      col.b = texture2D(u_texture, uv - vec2( amt, 0.0)).b;
      col.a = 1.0;
      if (highlightOnly) {
        // Only apply aberration on bright areas (highlights), avoids muddy look
        float luma = dot(col.rgb, vec3(0.299, 0.587, 0.114));
        vec4  base = texture2D(u_texture, uv);
        col  = mix(base, col, smoothstep(0.5, 0.85, luma));
      }
      return col;
    }

    /* Horizontal scanline darkening */
    float scanline(vec2 uv, float strength) {
      float line = sin(uv.y * u_resolution.y * 0.5 * 3.14159);
      return 1.0 - strength * (0.5 - 0.5 * line) * (0.5 - 0.5 * line);
    }

    /* Radial vignette */
    float vignette(vec2 uv, float strength) {
      vec2 c = (uv - 0.5) * 2.0;
      return 1.0 - dot(c, c) * strength;
    }

    /* Ripple displacement from a center point — single ring */
    vec2 rippleDisplace(vec2 uv, vec2 origin, float time, float amp, float freq, float phase) {
      vec2  d    = uv - origin;
      float dist = length(d) + 0.0001;
      float wave = sin(dist * freq - time * 3.0 + phase) * amp / dist;
      // Edge attenuation — reduce displacement near borders
      float edge = smoothstep(0.0, 0.12, min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)));
      return normalize(d) * wave * edge;
    }

    /* ─────────────────────────────────────────────────────────────────────
       MODE FUNCTIONS
       Each returns vec4 color for that UV.
       ───────────────────────────────────────────────────────────────────── */

    /* Idle — subtle grain + breathing vignette */
    vec4 modeIdle(vec2 uv) {
      vec4  col   = texture2D(u_texture, uv);
      float grain = hash(uv + fract(u_time * 0.07)) * 0.06 - 0.03;
      col.rgb    += grain;
      float vig   = vignette(uv, 0.35 + 0.1 * sin(u_time * 0.25));
      col.rgb    *= vig;
      // Mild scanline flicker
      col.rgb    *= scanline(uv, 0.03 + 0.02 * sin(u_time * 0.5));
      return col;
    }

    /* FIST — dither glitch mode */
    vec4 modeDither(vec2 uv) {
      // Velocity reduces quantization levels — more movement = coarser pixels
      float levels    = mix(8.0, 2.0, u_velocity * u_activationT);
      float pixelSize = mix(2.0, 6.0, u_velocity * 0.5) / min(u_resolution.x, u_resolution.y);

      // Pixelate
      vec2 pixUV  = floor(uv / pixelSize) * pixelSize;
      vec4 col    = texture2D(u_texture, pixUV);

      // Luma
      float luma  = dot(col.rgb, vec3(0.299, 0.587, 0.114));

      // Interleaved gradient noise — GLSL ES 1.00 safe alternative to Bayer matrix indexing
      // (mat4[runtime_index] is NOT allowed on macOS ANGLE)
      float thresh = fract(52.9829189 * fract(dot(floor(gl_FragCoord.xy), vec2(0.06711056, 0.00583715))));

      // Quantize
      float q     = floor(luma * levels + thresh) / levels;

      // Grain overlay
      float grain = hash(uv + fract(u_time * 0.31)) * 0.15;
      q           = clamp(q + grain * 0.15, 0.0, 1.0);

      // Column-shift glitch — random horizontal bands shift pixels left/right
      float glitchBand = hash(vec2(floor(uv.y * 60.0), u_time * 12.0));
      float glitchAmt  = step(0.92, glitchBand) * 0.03 * u_velocity;
      vec2  glitchUV   = vec2(uv.x + glitchAmt, uv.y);
      float q2         = dot(texture2D(u_texture, glitchUV).rgb, vec3(0.299, 0.587, 0.114));
      q = mix(q, q2, step(0.92, glitchBand));

      // Horizontal tear burst on velocity spikes
      float tear = step(0.96, hash(vec2(u_time * 30.0, 0.0))) * u_velocity;
      float tearY = hash(vec2(u_time * 7.0, 1.0));
      if (abs(uv.y - tearY) < 0.004 * u_velocity) q = mix(q, 1.0 - q, tear);

      return vec4(vec3(q), 1.0);
    }

    /* PEACE — VHS / CRT mode */
    vec4 modeVHS(vec2 uv) {
      // Barrel distortion first, then sample
      vec2 distUV    = barrel(uv, 0.06 + u_depth * 0.04);

      // Clamp to prevent sampling outside texture
      distUV         = clamp(distUV, 0.001, 0.999);

      // Chromatic aberration — luminance-weighted at highlights
      float aberAmt  = 0.003 + u_velocity * 0.004 + u_audioLevel * 0.003;
      vec4  col      = chromaticAberration(distUV, aberAmt, true);

      // Scanlines
      col.rgb        *= scanline(uv, 0.15);

      // Slow VHS noise drift
      float vhsNoise = noise2D(vec2(uv.y * 3.0, u_time * 0.8)) * 0.04;
      col.rgb        += vhsNoise * 0.5;

      // Random horizontal tracking glitch strips
      float glitchRand = hash(vec2(floor(uv.y * 120.0), floor(u_time * 6.0)));
      float glitchProb = 0.015 + u_audioLevel * 0.04;
      if (glitchRand > 1.0 - glitchProb) {
        float shift = (hash(vec2(glitchRand, u_time)) - 0.5) * 0.03;
        col = chromaticAberration(vec2(clamp(distUV.x + shift, 0.001, 0.999), distUV.y), aberAmt * 2.0, false);
        col.rgb *= 1.2;
      }

      // VHS color bleed — subtle warm tint
      col.r  *= 1.04;
      col.b  *= 0.92;

      // CRT vignette (harder edge than idle)
      col.rgb *= vignette(uv, 0.6);

      return col;
    }

    /* POINT — spotlight mode */
    vec4 modeSpotlight(vec2 uv) {
      vec4 col      = texture2D(u_texture, uv);

      // Convert fingertip from normalized (TL origin) to WebGL UV (BL origin)
      vec2 spotPos  = vec2(1.0 - u_fingerPos.x, u_fingerPos.y);

      float dist    = length(uv - spotPos);

      // Radius driven by depth — closer hand = wider spotlight
      float radius  = 0.18 + u_depth * 0.12;

      // Organic noisy feathering — prevents mechanical circle look
      float noiseAmt = fbm(uv * 4.0 + u_time * 0.3) * 0.04;
      float mask    = smoothstep(radius + 0.06 + noiseAmt, radius - 0.06, dist);

      // Dark surround — 0.08 multiply in shadow areas
      float dark    = 0.08;
      col.rgb       = mix(col.rgb * dark, col.rgb, mask);

      // Subtle bloom ring around spotlight edge
      float bloom   = smoothstep(radius + 0.1, radius, dist) - smoothstep(radius, radius - 0.04, dist);
      col.rgb      += bloom * 0.25 * u_activationT;

      return col;
    }

    /* OPEN HAND — water ripple mode */
    vec4 modeRipple(vec2 uv) {
      // Palm center in WebGL UV space
      vec2 palm     = vec2(1.0 - u_palmPos.x, u_palmPos.y);

      float amp     = 0.006 + u_velocity * 0.008;
      float freq    = 18.0 + u_depth * 12.0;

      // 3 rings unrolled (GLSL ES 1.00 doesn't allow non-constant break conditions)
      vec2 disp = vec2(0.0);
      disp += rippleDisplace(uv, palm, u_time, amp, freq, 0.0);
      disp += rippleDisplace(uv, palm, u_time, amp * 0.75, freq, 2.094);
      // Third ring only on MED/HIGH quality
      if (u_qualityTier >= 1.0) {
        disp += rippleDisplace(uv, palm, u_time, amp * 0.5, freq, 4.189);
      }

      // Audio adds high-frequency secondary shimmer
      if (u_audioLevel > 0.05) {
        float shimmerAmp  = u_audioLevel * 0.003;
        float shimmerFreq = 55.0;
        disp += rippleDisplace(uv, palm, u_time * 1.6, shimmerAmp, shimmerFreq, 0.0);
      }

      vec2 rippleUV = clamp(uv + disp * u_activationT, 0.001, 0.999);
      vec4 col      = texture2D(u_texture, rippleUV);

      // Refractive shimmer — tint displaced areas slightly
      float shimmerMask = length(disp) * 40.0;
      col.rgb          += shimmerMask * vec3(0.02, 0.04, 0.06);

      return col;
    }

    /* ─────────────────────────────────────────────────────────────────────
       COMPOSITOR — main()
       ───────────────────────────────────────────────────────────────────── */
    void main() {
      // 1. Always-on global mild barrel lens distortion
      vec2 uv       = barrel(v_uv, u_lensDistort);
      uv            = clamp(uv, 0.001, 0.999);

      // 2. Sample each mode at the distorted UV
      vec4 cIdle    = modeIdle(uv);
      vec4 cDither  = modeDither(uv);
      vec4 cVHS     = modeVHS(uv);
      vec4 cSpot    = modeSpotlight(uv);
      vec4 cRipple  = modeRipple(uv);

      // 3. Mix by blend weights (always sum to 1.0)
      vec4 color    = cIdle    * u_idleWeight
                    + cDither  * u_blendWeights.x  // fist
                    + cVHS     * u_blendWeights.y  // peace
                    + cSpot    * u_blendWeights.z  // point
                    + cRipple  * u_blendWeights.w; // open

      // 4. Activation ramp — scale effect intensity in/out on hand appear
      //    Idle is always visible; active modes scale with u_activationT
      float activeContrib = u_blendWeights.x + u_blendWeights.y + u_blendWeights.z + u_blendWeights.w;
      if (activeContrib > 0.01) {
        vec4 baseColor = cIdle;
        color = mix(baseColor, color, u_activationT);
      }

      gl_FragColor = clamp(color, 0.0, 1.0);
    }
  `;

  /* ── Post-processing fragment shader (pass 2, FBO → screen) ─────────── */
  const FRAG_POST = `
    precision highp float;
    varying vec2 v_uv;

    uniform sampler2D u_scene;        // rendered scene from pass 1
    uniform float     u_time;
    uniform vec2      u_resolution;
    uniform float     u_bloomIntensity; // 0.3 base, higher in spotlight
    uniform float     u_vigStrength;
    uniform float     u_grainStrength;
    uniform float     u_qualityTier;
    uniform float     u_blendWeightSpot; // spotlight mode weight
    uniform float     u_blendWeightDith; // dither mode weight
    uniform float     u_blendWeightVHS;  // VHS mode weight

    float hash(vec2 p) {
      p = fract(p * vec2(127.619, 311.712));
      p += dot(p, p + 19.31);
      return fract(p.x * p.y);
    }

    /* Box blur — fixed iteration count (GLSL ES 1.00 requires constant loop bounds) */
    vec4 boxBlur(sampler2D tex, vec2 uv, vec2 dir, float rad) {
      vec4  sum      = vec4(0.0);
      float total    = 0.0;
      vec2  texelSize = dir / u_resolution;
      // Fixed 13-tap blur (−16 to +6). Iterations outside [-rad, rad] are skipped.
      for (int i = -6; i <= 6; i++) {
        float fi = float(i);
        if (fi >= -rad && fi <= rad) {
          sum   += texture2D(tex, clamp(uv + texelSize * fi, 0.0, 1.0));
          total += 1.0;
        }
      }
      return sum / max(total, 1.0);
    }

    void main() {
      vec4 scene = texture2D(u_scene, v_uv);

      /* ── Bloom / Glow ─────────────────────────────────────────────── */
      // Threshold: only bright areas contribute to bloom
      float luma  = dot(scene.rgb, vec3(0.299, 0.587, 0.114));
      float thresh = 0.6;
      vec4  bright = scene * max(0.0, luma - thresh) / (1.0 - thresh + 0.001);

      // Number of blur passes scales with quality tier
      int passes = 1;
      if (u_qualityTier >= 1.0) passes = 2;
      if (u_qualityTier >= 2.0) passes = 3;

      // Approximation: 3 box-blur passes at increasing radii
      vec4 bloom = bright;
      float blurR = 6.0;
      for (int p = 0; p < 3; p++) {
        if (p >= passes) break;
        bloom = boxBlur(u_scene, v_uv, vec2(blurR, 0.0), blurR);
        bloom = boxBlur(u_scene, v_uv, vec2(0.0, blurR), blurR);
        blurR *= 2.0;
      }
      bloom = max(bloom - vec4(thresh), vec4(0.0));

      // Boost bloom intensity in spotlight mode
      float bloomBoost = 1.0 + u_blendWeightSpot * 1.3;
      vec4 color = scene + bloom * u_bloomIntensity * bloomBoost;

      /* ── Vignette ──────────────────────────────────────────────────── */
      vec2  c   = (v_uv - 0.5) * 2.0;
      // Breathing in idle, harder in VHS, suppressed in spotlight
      float vigMod = 1.0 - u_blendWeightSpot * 0.8 + u_blendWeightVHS * 0.2;
      float vigBreath = 1.0 + 0.08 * sin(u_time * 0.25);
      float vig = 1.0 - dot(c, c) * u_vigStrength * vigMod * vigBreath;
      color.rgb *= clamp(vig, 0.0, 1.0);

      /* ── Film Grain ────────────────────────────────────────────────── */
      // Grain resolution scales with quality tier
      float grainScale = 1.0 - u_qualityTier * 0.15;
      float grain = hash(v_uv * grainScale + fract(u_time * 0.1)) * 2.0 - 1.0;
      float grainAmt = u_grainStrength + u_blendWeightDith * 0.15;
      color.rgb += grain * grainAmt;

      /* ── Color Grade ──────────────────────────────────────────────── */
      // Subtle S-curve contrast
      color.rgb = color.rgb * (color.rgb * (2.51) + 0.03) / (color.rgb * (2.43) + 0.59);
      // Slightly raised blacks (cinematic look)
      color.rgb = color.rgb * 0.95 + 0.02;
      // Mild cool-to-warm tint — unify all modes visually
      color.r  *= 1.01;
      color.b  *= 0.98;

      gl_FragColor = clamp(color, 0.0, 1.0);
    }
  `;

  return { VERT_MAIN, VERT_POST, FRAG_MAIN, FRAG_POST };
})();

/* ════════════════════════════════════════════════════════════════════════════
   3. WEBGL ENGINE
   Context creation, shader compilation, FBO setup, texture management,
   uniform batching. Two-pass pipeline: main shaders → FBO → post-process.
   ════════════════════════════════════════════════════════════════════════════ */
const WebGLEngine = (() => {
  let gl = null;
  let mainProgram = null;
  let postProgram = null;
  let fbo = null;
  let fboTexture = null;
  let videoTexture = null;
  let quadBuffer = null;

  /* Uniform location caches */
  let mainUniforms = {};
  let postUniforms  = {};

  /* Current uniform values (smoothed before upload) */
  const uniforms = {
    blendWeights: [0, 0, 0, 0],  // [fist, peace, point, open]
    idleWeight: 1.0,
    fingerPos: [0.5, 0.5],
    palmPos: [0.5, 0.5],
    velocity: 0.0,
    audioLevel: 0.0,
    depth: 0.5,
    qualityTier: 2.0,
    activationT: 0.0,
  };

  function compileShader(type, src) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      const typeName = type === gl.VERTEX_SHADER ? 'VERTEX' : 'FRAGMENT';
      DebugLog.error(`Shader compile (${typeName})`, info);
      console.error(`Shader compile error (${typeName}):`, info);
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  function createProgram(vertSrc, fragSrc) {
    const vert = compileShader(gl.VERTEX_SHADER, vertSrc);
    const frag = compileShader(gl.FRAGMENT_SHADER, fragSrc);
    if (!vert || !frag) {
      DebugLog.error('createProgram', 'Shader compilation failed (vert=' + !!vert + ' frag=' + !!frag + ')');
      return null;
    }
    const program = gl.createProgram();
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      DebugLog.error('Program link', gl.getProgramInfoLog(program));
      return null;
    }
    return program;
  }

  function cacheUniforms(program, names) {
    const cache = {};
    for (const name of names) {
      cache[name] = gl.getUniformLocation(program, name);
    }
    return cache;
  }

  function createFBO(w, h) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { fb, texture };
  }

  function init(canvasEl, fallback2D) {
    // Try WebGL2 then WebGL1
    gl = canvasEl.getContext('webgl2') || canvasEl.getContext('webgl');
    if (!gl) { fallback2D(); return false; }

    // Compile programs
    mainProgram = createProgram(Shaders.VERT_MAIN, Shaders.FRAG_MAIN);
    postProgram = createProgram(Shaders.VERT_POST, Shaders.FRAG_POST);

    if (!mainProgram || !postProgram) {
      DebugLog.error('WebGLEngine.init', 'Shader programs failed to compile — falling back to 2D');
      fallback2D();
      gl = null;
      return false;
    }

    // Full-screen quad
    quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,   1, -1,   -1, 1,
       1, -1,   1,  1,   -1, 1,
    ]), gl.STATIC_DRAW);

    // Cache uniform locations
    mainUniforms = cacheUniforms(mainProgram, [
      'u_texture','u_time','u_delta','u_resolution',
      'u_blendWeights','u_idleWeight',
      'u_fingerPos','u_palmPos',
      'u_velocity','u_audioLevel','u_depth',
      'u_qualityTier','u_activationT','u_lensDistort',
    ]);
    postUniforms = cacheUniforms(postProgram, [
      'u_scene','u_time','u_resolution',
      'u_bloomIntensity','u_vigStrength','u_grainStrength',
      'u_qualityTier','u_blendWeightSpot','u_blendWeightDith','u_blendWeightVHS',
    ]);

    // Video texture
    videoTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, videoTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Initial FBO
    const { width, height } = canvasEl;
    const fb = createFBO(width, height);
    fbo = fb.fb;
    fboTexture = fb.texture;

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    return true;
  }

  function resizeFBO(w, h) {
    if (!gl) return;
    if (fbo) gl.deleteFramebuffer(fbo);
    if (fboTexture) gl.deleteTexture(fboTexture);
    const fb = createFBO(w, h);
    fbo = fb.fb;
    fboTexture = fb.texture;
  }

  function bindQuad(program) {
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    const loc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  }

  function uploadVideoFrame(videoEl) {
    if (!gl || !videoEl || videoEl.readyState < 2) return;
    gl.bindTexture(gl.TEXTURE_2D, videoTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, videoEl);
  }

  function render(videoEl, time, delta, w, h, blendWeights, idleWeight, activationT, velocity, audioLevel, depth, qualityTier) {
    if (!gl) return;

    uploadVideoFrame(videoEl);

    const dpr    = window.devicePixelRatio || 1;

    /* ── Pass 1: main shader → FBO ──────────────────────────────────── */
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(mainProgram);
    bindQuad(mainProgram);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, videoTexture);
    gl.uniform1i(mainUniforms.u_texture, 0);

    gl.uniform1f(mainUniforms.u_time,    time);
    gl.uniform1f(mainUniforms.u_delta,   delta);
    gl.uniform2f(mainUniforms.u_resolution, w, h);

    // Blend weights — smoothed before upload
    gl.uniform4f(mainUniforms.u_blendWeights,
      blendWeights[G.FIST],
      blendWeights[G.PEACE],
      blendWeights[G.POINT],
      blendWeights[G.OPEN]
    );
    gl.uniform1f(mainUniforms.u_idleWeight, idleWeight);

    gl.uniform2fv(mainUniforms.u_fingerPos, uniforms.fingerPos);
    gl.uniform2fv(mainUniforms.u_palmPos,   uniforms.palmPos);

    gl.uniform1f(mainUniforms.u_velocity,     velocity);
    gl.uniform1f(mainUniforms.u_audioLevel,   audioLevel);
    gl.uniform1f(mainUniforms.u_depth,        depth);
    gl.uniform1f(mainUniforms.u_qualityTier,  qualityTier);
    gl.uniform1f(mainUniforms.u_activationT,  activationT);
    gl.uniform1f(mainUniforms.u_lensDistort,  CONFIG.LENS_DISTORTION);

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    /* ── Pass 2: FBO → screen post-processing ───────────────────────── */
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);

    gl.useProgram(postProgram);
    bindQuad(postProgram);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, fboTexture);
    gl.uniform1i(postUniforms.u_scene, 0);

    gl.uniform1f(postUniforms.u_time,        time);
    gl.uniform2f(postUniforms.u_resolution,  w, h);
    gl.uniform1f(postUniforms.u_bloomIntensity, CONFIG.BLOOM_INTENSITY);
    gl.uniform1f(postUniforms.u_vigStrength,    CONFIG.VIGNETTE_STRENGTH);
    gl.uniform1f(postUniforms.u_grainStrength,  CONFIG.GRAIN_STRENGTH);
    gl.uniform1f(postUniforms.u_qualityTier,    qualityTier);
    gl.uniform1f(postUniforms.u_blendWeightSpot, blendWeights[G.POINT]);
    gl.uniform1f(postUniforms.u_blendWeightDith, blendWeights[G.FIST]);
    gl.uniform1f(postUniforms.u_blendWeightVHS,  blendWeights[G.PEACE]);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  return { init, resizeFBO, render, uniforms };
})();

/* ════════════════════════════════════════════════════════════════════════════
   5. TRACKING ENGINE
   MediaPipe Hands initialization, landmark processing, EMA velocity,
   depth estimation, confidence weighting. Supports two-hand tracking.
   All outputs in normalized 0–1.
   ════════════════════════════════════════════════════════════════════════════ */
const TrackingEngine = (() => {
  let hands = null;

  /* Per-hand state factory */
  function makeHandState() {
    return {
      fingerPos:    { x: 0.5, y: 0.5 },
      palmPos:      { x: 0.5, y: 0.5 },
      rawLandmarks: null,
      confidence:   0.0,
      velocity:     0.0,
      depth:        0.5,
      hasHand:      false,
      lastSeenMs:   0,
      handedness:   'unknown', // 'Left' or 'Right'
      /* Internal EMA accumulators */
      _prevCentroid: { x: 0.5, y: 0.5 },
      _velEMA:       0.0,
      _smoothDepth:  0.5,
    };
  }

  /* Multi-hand state: array of 2 hand states */
  const handStates = [makeHandState(), makeHandState()];

  /* Legacy single-hand alias (dominant hand = highest confidence) */
  const state = {
    get fingerPos()    { return getDominant().fingerPos; },
    get palmPos()      { return getDominant().palmPos; },
    get rawLandmarks() { return getDominant().rawLandmarks; },
    get confidence()   { return getDominant().confidence; },
    get velocity()     { return getDominant().velocity; },
    get depth()        { return getDominant().depth; },
    get hasHand()      { return handStates[0].hasHand || handStates[1].hasHand; },
    get lastSeenMs()   { return Math.max(handStates[0].lastSeenMs, handStates[1].lastSeenMs); },
  };

  function getDominant() {
    if (!handStates[1].hasHand) return handStates[0];
    if (!handStates[0].hasHand) return handStates[1];
    return handStates[0].confidence >= handStates[1].confidence ? handStates[0] : handStates[1];
  }

  function init() {
    hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: CONFIG.MIN_DETECTION_CONFIDENCE,
      minTrackingConfidence: 0.55,
    });

    hands.onResults(onResults);
    return hands;
  }

  function processHand(hs, lm, confidence, handednessLabel) {
    hs.rawLandmarks = lm;
    hs.confidence   = confidence;
    hs.hasHand      = true;
    hs.lastSeenMs   = performance.now();
    hs.handedness   = handednessLabel;

    // ── Index fingertip (landmark 8) ──────────────────────────────────────
    const tip = lm[8];
    const mirroredTipX = 1.0 - tip.x;

    hs.fingerPos.x += (mirroredTipX - hs.fingerPos.x) * CONFIG.POSITION_LERP;
    hs.fingerPos.y += (tip.y        - hs.fingerPos.y) * CONFIG.POSITION_LERP;

    // ── Palm center ──────────────────────────────────────────────────────
    const palmLM = [lm[0], lm[5], lm[9], lm[13], lm[17]];
    let palmX = 0, palmY = 0;
    for (const p of palmLM) { palmX += p.x; palmY += p.y; }
    palmX /= palmLM.length;
    palmY /= palmLM.length;
    const mirroredPalmX = 1.0 - palmX;

    hs.palmPos.x += (mirroredPalmX - hs.palmPos.x) * CONFIG.POSITION_LERP;
    hs.palmPos.y += (palmY         - hs.palmPos.y) * CONFIG.POSITION_LERP;

    // ── Velocity ──────────────────────────────────────────────────────────
    const centroidX = (hs.fingerPos.x + hs.palmPos.x) * 0.5;
    const centroidY = (hs.fingerPos.y + hs.palmPos.y) * 0.5;
    const dx = centroidX - hs._prevCentroid.x;
    const dy = centroidY - hs._prevCentroid.y;
    const rawVel = Math.sqrt(dx * dx + dy * dy) * confidence;
    hs._prevCentroid = { x: centroidX, y: centroidY };

    hs._velEMA = hs._velEMA + (rawVel - hs._velEMA) * CONFIG.VELOCITY_EMA;
    hs.velocity = Math.min(hs._velEMA * 12.0, 1.0);

    // ── Depth ──────────────────────────────────────────────────────────────
    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    for (const p of lm) {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    }
    const bboxArea  = (maxX - minX) * (maxY - minY);
    hs._smoothDepth += (bboxArea - hs._smoothDepth) * 0.15;
    hs.depth        = Math.min(hs._smoothDepth * 4.0, 1.0);
  }

  function onResults(results) {
    const numHands = results.multiHandLandmarks ? results.multiHandLandmarks.length : 0;

    // Process detected hands
    for (let i = 0; i < Math.min(numHands, 2); i++) {
      const lm         = results.multiHandLandmarks[i];
      const confidence  = results.multiHandedness?.[i]?.score ?? 0.85;
      const label       = results.multiHandedness?.[i]?.label ?? 'unknown';
      processHand(handStates[i], lm, confidence, label);
    }

    // Mark undetected hand slots as lost
    for (let i = numHands; i < 2; i++) {
      handStates[i].hasHand = false;
    }
  }

  async function send(videoEl) {
    if (!hands || !videoEl) return;
    await hands.send({ image: videoEl });
  }

  return { init, send, state, handStates, getDominant };
})();

/* ════════════════════════════════════════════════════════════════════════════
   6. GESTURE ENGINE
   Finger extension classifier → hysteresis ring buffer → stableGesture →
   blend weight interpolation with smoothstep ramping.
   ════════════════════════════════════════════════════════════════════════════ */
const GestureEngine = (() => {
  /* Public blend weights — always sum to 1.0 */
  const blendWeights = [0, 0, 0, 0, 0]; // indexed by G.*
  blendWeights[G.IDLE] = 1.0;

  /* Target weights for smooth interpolation */
  const targetWeights = [0, 0, 0, 0, 0];
  targetWeights[G.IDLE] = 1.0;

  let stableGesture = G.IDLE;
  let rawGesture    = G.IDLE;

  /* Hysteresis ring buffer */
  const historyBuf = new Array(CONFIG.HYSTERESIS_BUFFER).fill(G.IDLE);
  let historyIdx   = 0;

  /* Blend ramp speed: fraction per ms */
  const RAMP_SPEED = 1.0 / CONFIG.BLEND_RAMP_MS;

  /* Classify gesture from landmarks.
     Returns G.* constant.                                                */
  function classify(lm) {
    if (!lm) return G.IDLE;

    // Each finger: tip, dip, pip, mcp (knuckle)
    // Extended if tip is farther from wrist than pip
    const WRIST = lm[0];

    function dist2(a, b) {
      const dx = a.x - b.x, dy = a.y - b.y;
      return Math.sqrt(dx * dx + dy * dy);
    }

    // For each finger: extended when tip is above (lower y) pip in image space
    // More robust: tip-to-wrist distance > mcp-to-wrist distance
    function fingerExtended(tipIdx, pipIdx, mcpIdx) {
      const tipDist = dist2(lm[tipIdx], WRIST);
      const mcpDist = dist2(lm[mcpIdx], WRIST);
      return tipDist > mcpDist * 1.15;
    }

    // Thumb: uses angle at CMC joint (landmark 1) — compare x distance
    function thumbExtended() {
      // Simple: thumb tip x vs thumb mcp x (mirrored hand-agnostic)
      return Math.abs(lm[4].x - lm[2].x) > 0.08;
    }

    const index  = fingerExtended(8,  6,  5);
    const middle = fingerExtended(12, 10, 9);
    const ring   = fingerExtended(16, 14, 13);
    const pinky  = fingerExtended(20, 18, 17);
    const thumb  = thumbExtended();

    const extCount = [index, middle, ring, pinky, thumb].filter(Boolean).length;

    if (extCount === 0)                               return G.FIST;
    if (extCount >= 4)                                return G.OPEN;
    if (index && middle && !ring && !pinky)            return G.PEACE;
    if (index && !middle && !ring && !pinky)           return G.POINT;
    if (extCount >= 3)                                return G.OPEN; // relaxed open

    return G.IDLE;
  }

  function update(landmarks, deltaMs) {
    // Classify current frame
    rawGesture = landmarks ? classify(landmarks) : G.IDLE;

    // Push into hysteresis ring buffer
    historyBuf[historyIdx] = rawGesture;
    historyIdx = (historyIdx + 1) % CONFIG.HYSTERESIS_BUFFER;

    // Vote count
    const votes = new Array(5).fill(0);
    for (const g of historyBuf) votes[g]++;

    // Only switch if a gesture wins >= threshold votes
    let bestGesture = stableGesture;
    let bestVotes   = votes[stableGesture];
    for (let i = 0; i < 5; i++) {
      if (votes[i] > bestVotes) { bestVotes = votes[i]; bestGesture = i; }
    }
    if (bestVotes >= CONFIG.HYSTERESIS_THRESHOLD) {
      stableGesture = bestGesture;
    }

    // Update target weights — target = 1 for stable, 0 for others
    for (let i = 0; i < 5; i++) {
      targetWeights[i] = (i === stableGesture) ? 1.0 : 0.0;
    }

    // Interpolate blend weights toward targets using smoothstep-like ramping
    const step = RAMP_SPEED * deltaMs;
    let total  = 0.0;
    for (let i = 0; i < 5; i++) {
      blendWeights[i] += (targetWeights[i] - blendWeights[i]) * Math.min(step * 3.0, 1.0);
      blendWeights[i]  = Math.max(0.0, Math.min(1.0, blendWeights[i]));
      total           += blendWeights[i];
    }
    // Normalize so weights always sum to 1.0
    if (total > 0.001) {
      for (let i = 0; i < 5; i++) blendWeights[i] /= total;
    }
  }

  return {
    update,
    get stableGesture() { return stableGesture; },
    get rawGesture()    { return rawGesture; },
    blendWeights,
  };
})();

/* ════════════════════════════════════════════════════════════════════════════
   7. PARTICLE SYSTEM
   Object-pooled trail particles drawn on the 2D overlay canvas.
   Additive blending, velocity-based trail length, fingertip orb.
   ════════════════════════════════════════════════════════════════════════════ */
const ParticleSystem = (() => {
  let ctx         = null;
  let canvasEl    = null;
  let pool        = [];
  let activeCount = 0;
  let maxParticles = CONFIG.PARTICLE_POOL.HIGH;

  /* Particle object template */
  function makeParticle() {
    return {
      active: false,
      x: 0, y: 0,
      vx: 0, vy: 0,
      radius: 0,
      alpha: 0,
      life: 0,
      maxLife: 0,
      r: 255, g: 255, b: 255,
    };
  }

  function init(overlayCanvas) {
    canvasEl = overlayCanvas;
    ctx      = canvasEl.getContext('2d');
    // Pre-allocate pool
    pool = Array.from({ length: CONFIG.PARTICLE_POOL.HIGH }, makeParticle);
  }

  function setMaxParticles(max) {
    maxParticles = max;
    activeCount  = Math.min(activeCount, max);
  }

  /* Spawn a new particle at fingertip position */
  function spawnParticle(fx, fy, velX, velY, velocity) {
    // Find first inactive slot
    for (let i = 0; i < maxParticles; i++) {
      const p = pool[i];
      if (!p.active) {
        p.active   = true;
        p.x        = fx;
        p.y        = fy;
        // Slightly randomize velocity direction
        const angle = Math.random() * Math.PI * 2;
        const speed = (0.5 + Math.random() * 0.5) * (0.5 + velocity * 2.5);
        p.vx       = -velX * 0.4 + Math.cos(angle) * speed;
        p.vy       = -velY * 0.4 + Math.sin(angle) * speed;
        p.radius   = 3 + Math.random() * 4;
        p.alpha    = 0.8 + Math.random() * 0.2;
        p.life     = 0;
        const maxLifeMs = CONFIG.PARTICLE_LIFETIME_MS * (0.7 + Math.random() * 0.6)
                        * (1.0 + velocity * 0.8);
        p.maxLife  = maxLifeMs;
        // Neon color palette: cyan, magenta, white, teal
        const colors = [
          [0,   240, 255],  // cyan
          [220,  80, 255],  // magenta
          [255, 255, 255],  // white
          [80,  255, 200],  // teal
        ];
        const c    = colors[Math.floor(Math.random() * colors.length)];
        p.r = c[0]; p.g = c[1]; p.b = c[2];
        activeCount++;
        return;
      }
    }
  }

  /* Draw orb glow at fingertip position */
  function drawOrb(fx, fy, activationT) {
    if (!ctx || activationT < 0.01) return;

    // DPR-aware pixel coords
    const dpr = window.devicePixelRatio || 1;
    const px  = fx * canvasEl.width;
    const py  = fy * canvasEl.height;
    const r   = CONFIG.ORB_RADIUS * dpr;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter'; // additive blend

    const grad = ctx.createRadialGradient(px, py, 0, px, py, r * 2);
    grad.addColorStop(0,   `rgba(255,255,255,${0.95 * activationT})`);
    grad.addColorStop(0.3, `rgba(180,240,255,${0.6  * activationT})`);
    grad.addColorStop(0.7, `rgba(80, 200,255,${0.2  * activationT})`);
    grad.addColorStop(1,   'rgba(0,0,0,0)');

    ctx.beginPath();
    ctx.arc(px, py, r * 2, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.restore();
  }

  function beginFrame() {
    if (!ctx) return;
    const w = canvasEl.width;
    const h = canvasEl.height;
    ctx.clearRect(0, 0, w, h);
    ctx.globalCompositeOperation = 'source-over';
  }

  function update(deltaMs, fingerPos, velX, velY, velocity, activationT, emitting) {
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w   = canvasEl.width;
    const h   = canvasEl.height;

    // Emit new particles this frame (emission scales with activation)
    const emitRate = emitting ? Math.ceil((1 + velocity * 4) * activationT) : 0;
    const fx       = fingerPos.x * w;
    const fy       = fingerPos.y * h;

    for (let e = 0; e < emitRate; e++) {
      spawnParticle(fx, fy, velX * w, velY * h, velocity);
    }

    // Draw orb
    drawOrb(fingerPos.x, fingerPos.y, activationT);

    // Update and draw active particles
    ctx.globalCompositeOperation = 'lighter'; // additive blend for neon glow
    activeCount = 0;

    for (let i = 0; i < maxParticles; i++) {
      const p = pool[i];
      if (!p.active) continue;

      p.life += deltaMs;
      if (p.life >= p.maxLife) {
        p.active = false;
        continue;
      }

      activeCount++;
      const t     = p.life / p.maxLife;
      const alpha = p.alpha * (1 - t * t); // quadratic fade

      // Move particle
      p.x += p.vx * deltaMs * 0.04;
      p.y += p.vy * deltaMs * 0.04;

      // Draw glowing circle
      const pr   = p.radius * (1 - t * 0.5);
      const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, pr * 3);
      grad.addColorStop(0,   `rgba(${p.r},${p.g},${p.b},${alpha})`);
      grad.addColorStop(0.5, `rgba(${p.r},${p.g},${p.b},${alpha * 0.4})`);
      grad.addColorStop(1,   'rgba(0,0,0,0)');

      ctx.beginPath();
      ctx.arc(p.x, p.y, pr * 3, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
    }
  }

  return { init, beginFrame, update, setMaxParticles, get activeCount() { return activeCount; } };
})();

/* ════════════════════════════════════════════════════════════════════════════
   7b. SKELETON RENDERER
   Draws the MediaPipe 21-landmark hand skeleton as a glowing wireframe
   on the 2D overlay canvas. Supports two-hand rendering with per-finger
   color coding, depth-based sizing, and configurable appearance.
   ════════════════════════════════════════════════════════════════════════════ */
const SkeletonRenderer = (() => {
  let ctx      = null;
  let canvasEl = null;

  function init(overlayCanvas) {
    canvasEl = overlayCanvas;
    ctx      = canvasEl.getContext('2d');
  }

  /* Convert landmark (normalized 0–1) to canvas pixel coords (mirrored) */
  function toPixel(lm) {
    return {
      x: (1.0 - lm.x) * canvasEl.width,
      y: lm.y * canvasEl.height,
      z: lm.z || 0,
    };
  }

  /* Draw a single bone (line between two landmarks) with glow */
  function drawBone(p1, p2, color, lineWidth, opacity) {
    ctx.save();
    ctx.globalAlpha = opacity;
    ctx.globalCompositeOperation = 'lighter';

    // Outer glow
    ctx.strokeStyle = `rgba(${color[0]},${color[1]},${color[2]},0.3)`;
    ctx.lineWidth   = lineWidth * 3;
    ctx.lineCap     = 'round';
    ctx.shadowColor = `rgb(${color[0]},${color[1]},${color[2]})`;
    ctx.shadowBlur  = CONFIG.SKELETON_GLOW_BLUR;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();

    // Core line
    ctx.shadowBlur  = 0;
    ctx.strokeStyle = `rgba(${color[0]},${color[1]},${color[2]},0.9)`;
    ctx.lineWidth   = lineWidth;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();

    ctx.restore();
  }

  /* Draw a single joint (circle at a landmark) with glow */
  function drawJoint(p, color, radius, opacity, isTip) {
    ctx.save();
    ctx.globalAlpha = opacity;
    ctx.globalCompositeOperation = 'lighter';

    const r = isTip ? radius * 1.4 : radius;

    // Glow
    const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 3);
    grad.addColorStop(0,   `rgba(${color[0]},${color[1]},${color[2]},0.9)`);
    grad.addColorStop(0.4, `rgba(${color[0]},${color[1]},${color[2]},0.4)`);
    grad.addColorStop(1,   `rgba(${color[0]},${color[1]},${color[2]},0)`);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r * 3, 0, Math.PI * 2);
    ctx.fill();

    // Core dot
    ctx.fillStyle = `rgba(255,255,255,0.95)`;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r * 0.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  /* Render skeleton for a single hand */
  function renderHand(landmarks, handIndex, activationT) {
    if (!ctx || !landmarks || !CONFIG.SKELETON_VISIBLE) return;
    if (activationT < 0.01) return;

    const dpr       = window.devicePixelRatio || 1;
    const boneWidth = CONFIG.SKELETON_BONE_WIDTH * dpr;
    const jointSize = CONFIG.SKELETON_JOINT_SIZE * dpr;
    const opacity   = CONFIG.SKELETON_OPACITY * activationT;

    // Default uniform color per hand (left=cyan, right=magenta)
    const uniformColor = handIndex === 0
      ? [0, 255, 255]   // cyan
      : [255, 80, 255]; // magenta

    // Convert all landmarks to pixel coords
    const pixels = landmarks.map(toPixel);

    // Draw bones
    for (const [a, b] of HAND_CONNECTIONS) {
      const color = CONFIG.SKELETON_FINGER_COLORS
        ? getBoneColor(a, b)
        : uniformColor;

      // Depth-based thickness: closer landmarks → thicker
      const avgZ   = (pixels[a].z + pixels[b].z) * 0.5;
      const depthScale = 1.0 + Math.max(0, -avgZ) * 2.0;
      drawBone(pixels[a], pixels[b], color, boneWidth * depthScale, opacity);
    }

    // Draw joints
    const tipIndices = new Set([4, 8, 12, 16, 20]);
    for (let i = 0; i < landmarks.length; i++) {
      const finger = getLandmarkFinger(i);
      const color  = CONFIG.SKELETON_FINGER_COLORS
        ? FINGER_COLORS[finger]
        : uniformColor;

      const depthScale = 1.0 + Math.max(0, -pixels[i].z) * 2.0;
      drawJoint(pixels[i], color, jointSize * depthScale, opacity, tipIndices.has(i));
    }
  }

  /* Main update: draw skeletons for all active hands */
  function update(handStates, activationT) {
    if (!ctx || !CONFIG.SKELETON_VISIBLE) return;

    for (let i = 0; i < handStates.length; i++) {
      const hs = handStates[i];
      if (hs.hasHand && hs.rawLandmarks) {
        renderHand(hs.rawLandmarks, i, activationT);
      }
    }
  }

  return { init, update };
})();

/* ════════════════════════════════════════════════════════════════════════════
   8. AUDIO ENGINE
   M-key opt-in microphone. AnalyserNode → RMS audioLevel → peak detect.
   Feeds u_audioLevel uniform.
   ════════════════════════════════════════════════════════════════════════════ */
const AudioEngine = (() => {
  let audioCtx    = null;
  let analyser    = null;
  let dataArray   = null;
  let active      = false;
  let denied      = false;
  let peakFrames  = 0;

  const state = {
    audioLevel: 0.0,
    peak: false,
    micDotEl:  null,
    micLabelEl: null,
  };

  function init(micDotEl, micLabelEl) {
    state.micDotEl   = micDotEl;
    state.micLabelEl = micLabelEl;

    document.addEventListener('keydown', (e) => {
      if (e.key === 'm' || e.key === 'M') {
        if (!active && !denied) enableMic();
        else if (active) disableMic();
      }
    });
  }

  async function enableMic() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      audioCtx  = new (window.AudioContext || window.webkitAudioContext)();
      const src = audioCtx.createMediaStreamSource(stream);
      analyser  = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      src.connect(analyser);
      dataArray = new Uint8Array(analyser.frequencyBinCount);
      active    = true;

      state.micDotEl.className  = 'mic-dot-on';
      state.micLabelEl.textContent = 'M · mic on';
    } catch (err) {
      denied = true;
      state.micDotEl.className  = 'mic-dot-unavailable';
      state.micLabelEl.textContent = 'mic unavailable';
    }
  }

  function disableMic() {
    if (!audioCtx) return;
    audioCtx.close();
    audioCtx = analyser = dataArray = null;
    active   = false;
    state.micDotEl.className   = 'mic-dot-off';
    state.micLabelEl.textContent = 'M · mic';
  }

  function update() {
    if (!active || !analyser) {
      state.audioLevel = 0;
      state.peak       = false;
      return;
    }

    analyser.getByteFrequencyData(dataArray);

    // RMS of frequency data
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) sum += dataArray[i] * dataArray[i];
    const rms = Math.sqrt(sum / dataArray.length) / 255.0;

    // EMA smooth
    state.audioLevel += (rms - state.audioLevel) * CONFIG.AUDIO_EMA;
    state.audioLevel  = Math.min(state.audioLevel, 1.0);

    // Peak detection
    if (rms > CONFIG.AUDIO_PEAK_THRESHOLD) peakFrames = CONFIG.AUDIO_PEAK_FRAMES;
    state.peak = peakFrames > 0;
    if (peakFrames > 0) peakFrames--;
  }

  return { init, update, state };
})();

/* ════════════════════════════════════════════════════════════════════════════
   9. QUALITY MANAGER
   Rolling FPS monitor, adaptive tier stepping, Q-key manual cycle.
   ════════════════════════════════════════════════════════════════════════════ */
const QualityManager = (() => {
  let currentTier     = TIER.HIGH;
  let fpsWindow       = new Array(CONFIG.FPS_WINDOW).fill(CONFIG.TARGET_FPS);
  let windowIdx       = 0;
  let belowSeconds    = 0;
  let aboveSeconds    = 0;
  let qualityLabelEl  = null;

  const PARTICLE_POOLS = [CONFIG.PARTICLE_POOL.LOW, CONFIG.PARTICLE_POOL.MEDIUM, CONFIG.PARTICLE_POOL.HIGH];

  function init(labelEl) {
    qualityLabelEl = labelEl;
    document.addEventListener('keydown', (e) => {
      if (e.key === 'q' || e.key === 'Q') {
        currentTier = (currentTier + 1) % 3;
        apply();
      }
    });
    apply();
  }

  function apply() {
    ParticleSystem.setMaxParticles(PARTICLE_POOLS[currentTier]);
    if (qualityLabelEl) qualityLabelEl.textContent = `Q · ${TIER_NAMES[currentTier]}`;
  }

  function update(fps, deltaS) {
    fpsWindow[windowIdx] = fps;
    windowIdx = (windowIdx + 1) % CONFIG.FPS_WINDOW;

    const avgFPS = fpsWindow.reduce((a, b) => a + b) / CONFIG.FPS_WINDOW;
    const target = CONFIG.TARGET_FPS;

    if (avgFPS < target * CONFIG.FPS_DOWN_THRESHOLD) {
      belowSeconds += deltaS;
      aboveSeconds  = 0;
      if (belowSeconds >= CONFIG.FPS_DOWN_DURATION_S && currentTier > TIER.LOW) {
        currentTier--;
        belowSeconds = 0;
        apply();
      }
    } else if (avgFPS > target * CONFIG.FPS_UP_THRESHOLD) {
      aboveSeconds += deltaS;
      belowSeconds  = 0;
      if (aboveSeconds >= CONFIG.FPS_UP_DURATION_S && currentTier < TIER.HIGH) {
        currentTier++;
        aboveSeconds = 0;
        apply();
      }
    } else {
      belowSeconds = 0;
      aboveSeconds = 0;
    }

    return currentTier;
  }

  return { init, update, get currentTier() { return currentTier; } };
})();

/* ════════════════════════════════════════════════════════════════════════════
   9b. SETTINGS PANEL
   Collapsible sidebar with real-time sliders and toggles wired to CONFIG.
   Persists to localStorage. Toggle with S key or gear button.
   ════════════════════════════════════════════════════════════════════════════ */
const SettingsPanel = (() => {
  let panelEl     = null;
  let gearBtn     = null;
  let isOpen      = false;
  const STORAGE_KEY = 'xray-hand-settings';

  /* Slider/toggle definitions: { id, configKey, transform, display } */
  const sliderDefs = [
    { id: 'slider-bone-width',      valId: 'val-bone-width',      key: 'SKELETON_BONE_WIDTH',   transform: v => parseFloat(v), display: v => v },
    { id: 'slider-joint-size',      valId: 'val-joint-size',      key: 'SKELETON_JOINT_SIZE',   transform: v => parseFloat(v), display: v => v },
    { id: 'slider-skeleton-opacity',valId: 'val-skeleton-opacity', key: 'SKELETON_OPACITY',      transform: v => parseFloat(v) / 100, display: v => Math.round(v * 100) + '%' },
    { id: 'slider-bloom',           valId: 'val-bloom',           key: 'BLOOM_INTENSITY',       transform: v => parseFloat(v) / 100, display: v => Math.round(v * 100) + '%' },
    { id: 'slider-vignette',        valId: 'val-vignette',        key: 'VIGNETTE_STRENGTH',     transform: v => parseFloat(v) / 100, display: v => Math.round(v * 100) + '%' },
    { id: 'slider-grain',           valId: 'val-grain',           key: 'GRAIN_STRENGTH',        transform: v => parseFloat(v) / 100, display: v => Math.round(v * 100) + '%' },
    { id: 'slider-lens',            valId: 'val-lens',            key: 'LENS_DISTORTION',       transform: v => parseFloat(v) / 100, display: v => (v * 100).toFixed(1) + '%' },
    { id: 'slider-smoothing',       valId: 'val-smoothing',       key: 'POSITION_LERP',         transform: v => parseFloat(v) / 100, display: v => Math.round(v * 100) + '%' },
    { id: 'slider-confidence',      valId: 'val-confidence',      key: 'MIN_DETECTION_CONFIDENCE', transform: v => parseFloat(v) / 100, display: v => Math.round(v * 100) + '%' },
  ];

  const toggleDefs = [
    { id: 'toggle-skeleton',      key: 'SKELETON_VISIBLE' },
    { id: 'toggle-finger-colors', key: 'SKELETON_FINGER_COLORS' },
    { id: 'toggle-two-hands',     key: 'TWO_HAND_MODE' },
  ];

  function init() {
    panelEl = document.getElementById('settings-panel');
    gearBtn = document.getElementById('gear-btn');

    if (!panelEl || !gearBtn) return;

    // Gear button click
    gearBtn.addEventListener('click', toggle);

    // Close button
    const closeBtn = document.getElementById('panel-close-btn');
    if (closeBtn) closeBtn.addEventListener('click', () => setOpen(false));

    // S key toggle
    document.addEventListener('keydown', (e) => {
      if (e.key === 's' || e.key === 'S') toggle();
    });

    // H key toggles skeleton
    document.addEventListener('keydown', (e) => {
      if (e.key === 'h' || e.key === 'H') {
        CONFIG.SKELETON_VISIBLE = !CONFIG.SKELETON_VISIBLE;
        const el = document.getElementById('toggle-skeleton');
        if (el) el.checked = CONFIG.SKELETON_VISIBLE;
        save();
      }
    });

    // Wire sliders
    for (const def of sliderDefs) {
      const slider = document.getElementById(def.id);
      const valEl  = document.getElementById(def.valId);
      if (!slider) continue;

      slider.addEventListener('input', () => {
        const raw = slider.value;
        CONFIG[def.key] = def.transform(raw);
        if (valEl) valEl.textContent = def.display(CONFIG[def.key]);
        save();
      });
    }

    // Wire toggles
    for (const def of toggleDefs) {
      const toggle = document.getElementById(def.id);
      if (!toggle) continue;

      toggle.addEventListener('change', () => {
        CONFIG[def.key] = toggle.checked;
        save();
      });
    }

    // Wire landmark data toggle
    const lmToggle = document.getElementById('toggle-landmark-data');
    const lmContainer = document.getElementById('landmark-data-container');
    if (lmToggle && lmContainer) {
      lmToggle.addEventListener('change', () => {
        lmContainer.style.display = lmToggle.checked ? 'block' : 'none';
      });
    }

    // Load saved settings
    load();
  }

  function toggle() {
    setOpen(!isOpen);
  }

  function setOpen(open) {
    isOpen = open;
    if (panelEl) {
      if (isOpen) panelEl.classList.add('open');
      else panelEl.classList.remove('open');
    }
    document.body.classList.toggle('panel-open', isOpen);
  }

  function save() {
    const data = {};
    for (const def of sliderDefs) data[def.key] = CONFIG[def.key];
    for (const def of toggleDefs) data[def.key] = CONFIG[def.key];
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); } catch (e) { /* ignore */ }
  }

  function load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const data = JSON.parse(raw);

      for (const def of sliderDefs) {
        if (def.key in data) {
          CONFIG[def.key] = data[def.key];
          const slider = document.getElementById(def.id);
          const valEl  = document.getElementById(def.valId);
          if (slider) {
            // Reverse transform to get slider value
            if (def.key === 'SKELETON_OPACITY' || def.key === 'BLOOM_INTENSITY' ||
                def.key === 'VIGNETTE_STRENGTH' || def.key === 'GRAIN_STRENGTH' ||
                def.key === 'LENS_DISTORTION' || def.key === 'POSITION_LERP' ||
                def.key === 'MIN_DETECTION_CONFIDENCE') {
              slider.value = CONFIG[def.key] * 100;
            } else {
              slider.value = CONFIG[def.key];
            }
          }
          if (valEl) valEl.textContent = def.display(CONFIG[def.key]);
        }
      }

      for (const def of toggleDefs) {
        if (def.key in data) {
          CONFIG[def.key] = data[def.key];
          const toggle = document.getElementById(def.id);
          if (toggle) toggle.checked = CONFIG[def.key];
        }
      }
    } catch (e) { /* ignore corrupt storage */ }
  }

  return { init, toggle, get isOpen() { return isOpen; } };
})();

/* ════════════════════════════════════════════════════════════════════════════
   9c. LANDMARK DISPLAY
   Real-time table showing finger extension angles and tip positions.
   Throttled to ~10 Hz to avoid DOM thrashing.
   ════════════════════════════════════════════════════════════════════════════ */
const LandmarkDisplay = (() => {
  let tbodyEl      = null;
  let lastUpdateMs = 0;
  const UPDATE_INTERVAL_MS = 100; // ~10 Hz

  /* Finger definitions: name, tip landmark, pip landmark, mcp landmark */
  const FINGERS = [
    { name: 'Thumb',  tip: 4,  pip: 3,  mcp: 2  },
    { name: 'Index',  tip: 8,  pip: 6,  mcp: 5  },
    { name: 'Middle', tip: 12, pip: 10, mcp: 9  },
    { name: 'Ring',   tip: 16, pip: 14, mcp: 13 },
    { name: 'Pinky',  tip: 20, pip: 18, mcp: 17 },
  ];

  function init() {
    tbodyEl = document.getElementById('landmark-tbody');
  }

  /* Compute angle between three points (in degrees) */
  function angleDeg(a, b, c) {
    const ba = { x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) };
    const bc = { x: c.x - b.x, y: c.y - b.y, z: (c.z || 0) - (b.z || 0) };
    const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
    const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y + ba.z * ba.z);
    const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y + bc.z * bc.z);
    if (magBA < 0.001 || magBC < 0.001) return 0;
    const cosAngle = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
    return Math.round(Math.acos(cosAngle) * (180 / Math.PI));
  }

  function update(handStates) {
    if (!tbodyEl) return;
    const container = document.getElementById('landmark-data-container');
    if (!container || container.style.display === 'none') return;

    const now = performance.now();
    if (now - lastUpdateMs < UPDATE_INTERVAL_MS) return;
    lastUpdateMs = now;

    // Use dominant hand's landmarks
    let lm = null;
    for (const hs of handStates) {
      if (hs.hasHand && hs.rawLandmarks) { lm = hs.rawLandmarks; break; }
    }

    const rows = tbodyEl.querySelectorAll('tr');
    for (let i = 0; i < FINGERS.length && i < rows.length; i++) {
      const cells = rows[i].querySelectorAll('td');
      if (!lm) {
        cells[1].textContent = '—';
        cells[2].textContent = '—';
        cells[3].textContent = '—';
        cells[4].textContent = '—';
        continue;
      }

      const f   = FINGERS[i];
      const tip = lm[f.tip];
      const pip = lm[f.pip];
      const mcp = lm[f.mcp];

      const angle = angleDeg(tip, pip, mcp);
      cells[1].textContent = angle + '°';
      cells[2].textContent = (1.0 - tip.x).toFixed(2); // mirrored X
      cells[3].textContent = tip.y.toFixed(2);
      cells[4].textContent = (tip.z || 0).toFixed(3);
    }
  }

  return { init, update };
})();

/* ════════════════════════════════════════════════════════════════════════════
   10. UI LAYER
   Loading screen, gesture label, mic indicator, quality chip,
   debug overlay, activation hint, keyboard shortcuts.
   ════════════════════════════════════════════════════════════════════════════ */
const UILayer = (() => {
  let loadingEl    = null;
  let gestureEl    = null;
  let gestureText  = null;
  let hintEl       = null;
  let debugEl      = null;
  let debugVisible = false;

  let dbgFPS, dbgGesture, dbgConf, dbgWeights, dbgParticles, dbgQuality, dbgVelocity, dbgAudio;

  let hintTimer      = null;
  let hintShown      = false;
  let lastStableGesture = -1;

  function init() {
    loadingEl   = document.getElementById('loading-screen');
    gestureEl   = document.getElementById('gesture-label');
    gestureText = document.getElementById('gesture-text');
    hintEl      = document.getElementById('activation-hint');
    debugEl     = document.getElementById('debug-overlay');

    dbgFPS      = document.getElementById('dbg-fps');
    dbgGesture  = document.getElementById('dbg-gesture');
    dbgConf     = document.getElementById('dbg-conf');
    dbgWeights  = document.getElementById('dbg-weights');
    dbgParticles= document.getElementById('dbg-particles');
    dbgQuality  = document.getElementById('dbg-quality');
    dbgVelocity = document.getElementById('dbg-velocity');
    dbgAudio    = document.getElementById('dbg-audio');

    // D key toggles debug overlay
    document.addEventListener('keydown', (e) => {
      if (e.key === 'd' || e.key === 'D') {
        debugVisible = !debugVisible;
        debugEl.hidden = !debugVisible;
      }
    });

    // Schedule activation hint after ACTIVATION_HINT_DELAY_S
    hintTimer = setTimeout(() => {
      if (!TrackingEngine.state.hasHand) {
        hintEl.classList.add('visible');
        hintShown = true;
      }
    }, CONFIG.ACTIVATION_HINT_DELAY_S * 1000);
  }

  function hideLoading() {
    if (loadingEl) loadingEl.classList.add('hidden');
  }

  function onHandDetected() {
    if (hintShown) {
      hintEl.classList.remove('visible');
    }
    if (hintTimer) { clearTimeout(hintTimer); hintTimer = null; }
  }

  function updateGestureLabel(stable) {
    if (stable !== lastStableGesture) {
      lastStableGesture = stable;
      gestureText.style.opacity = '0';
      setTimeout(() => {
        gestureText.textContent = GESTURE_NAMES[stable];
        gestureText.style.opacity = '1';
      }, 150);
    }
  }

  function updateDebug(fps, stable, conf, weights, particles, tier, vel, audio) {
    if (!debugVisible) return;
    dbgFPS.textContent      = fps.toFixed(1);
    dbgGesture.textContent  = GESTURE_NAMES[stable];
    dbgConf.textContent     = (conf * 100).toFixed(0) + '%';
    dbgWeights.textContent  = weights.map(w => w.toFixed(2)).join(' ');
    dbgParticles.textContent= particles;
    dbgQuality.textContent  = TIER_NAMES[tier];
    dbgVelocity.textContent = vel.toFixed(3);
    dbgAudio.textContent    = audio.toFixed(3);
  }

  return { init, hideLoading, onHandDetected, updateGestureLabel, updateDebug };
})();

/* ════════════════════════════════════════════════════════════════════════════
   11. RENDER LOOP
   requestAnimationFrame loop. Orchestrates all engines each frame.
   Handles deltaTime, DPR-aware canvas sizing, activation ramp, and
   WebGL fallback (2D canvas mode).
   ════════════════════════════════════════════════════════════════════════════ */
const RenderLoop = (() => {
  let glCanvas     = null;
  let overlayCanvas = null;
  let webglAvail   = true;

  // Timing
  let lastTimestamp = 0;
  let totalTime     = 0;
  let frameCount    = 0;

  // Rolling FPS (60-frame window)
  const fpsBuffer = new Array(60).fill(60);
  let fpsBufIdx   = 0;
  let displayFPS  = 60;

  // Activation ramp state
  let activationT    = 0.0;    // 0–1 ramp value fed to shader
  let targetActivation = 0.0;  // 1 when hand present, 0 when lost
  let lostTrackingAt = 0;      // timestamp of last LOST_TRACKING

  // Smoothed shader uniforms (prevent frame-noise popping)
  let smoothVelocity = 0.0;
  let smoothAudio    = 0.0;
  let smoothDepth    = 0.5;
  let prevFingerX    = 0.5, prevFingerY = 0.5;

  // Velocity delta for particle system
  let velDeltaX = 0.0, velDeltaY = 0.0;

  // Flag: are we currently sending a frame to MediaPipe? (prevent overlap)
  let _mpBusy = false;

  function fallback2D() {
    webglAvail = false;
    DebugLog.log('Fallback2D: showing <video> element fullscreen as background');

    // Show the video element directly as the background (mirrored)
    const video = document.getElementById('webcam-video');
    if (video) {
      video.style.cssText = `
        position:fixed; top:0; left:0; width:100vw; height:100vh;
        object-fit:cover; z-index:0; opacity:1;
        transform:scaleX(-1); pointer-events:none;
      `;
    }
    // Hide gl-canvas since WebGL is non-functional
    const glc = document.getElementById('gl-canvas');
    if (glc) glc.style.display = 'none';
  }

  function resizeCanvases() {
    const dpr = window.devicePixelRatio || 1;
    const w   = window.innerWidth;
    const h   = window.innerHeight;

    glCanvas.width      = Math.round(w * dpr);
    glCanvas.height     = Math.round(h * dpr);
    glCanvas.style.width  = `${w}px`;
    glCanvas.style.height = `${h}px`;

    overlayCanvas.width  = Math.round(w * dpr);
    overlayCanvas.height = Math.round(h * dpr);
    overlayCanvas.style.width  = `${w}px`;
    overlayCanvas.style.height = `${h}px`;

    if (webglAvail) WebGLEngine.resizeFBO(glCanvas.width, glCanvas.height);
  }

  async function init() {
    DebugLog.log('RenderLoop.init() called');
    glCanvas      = document.getElementById('gl-canvas');
    overlayCanvas = document.getElementById('overlay-canvas');
    DebugLog.log('Canvas elements found: gl=' + !!glCanvas + ' overlay=' + !!overlayCanvas);

    // Init subsystems
    DebugLog.log('Step 0: Initializing UI subsystems...');
    try {
      UILayer.init();
      ParticleSystem.init(overlayCanvas);
      SkeletonRenderer.init(overlayCanvas);
      AudioEngine.init(
        document.getElementById('mic-dot'),
        document.getElementById('mic-label')
      );
      QualityManager.init(document.getElementById('quality-label'));
      SettingsPanel.init();
      LandmarkDisplay.init();
      DebugLog.log('Step 0: ✅ UI subsystems ready');
    } catch (err) {
      DebugLog.error('Step 0: UI subsystem init FAILED', err);
      return;
    }

    // Resize and observe
    resizeCanvases();
    new ResizeObserver(resizeCanvases).observe(document.body);

    // WebGL init
    DebugLog.log('Step 0b: Initializing WebGL...');
    try {
      webglAvail = WebGLEngine.init(glCanvas, fallback2D);
      DebugLog.log('Step 0b: ✅ WebGL available = ' + webglAvail);
    } catch (err) {
      DebugLog.error('Step 0b: WebGL init FAILED', err);
      webglAvail = false;
    }

    // ── Step 1: Open webcam via raw getUserMedia ────────────────────────
    DebugLog.log('Step 1: About to request camera access...');
    try {
      await WebcamSetup.init();
      DebugLog.log('Step 1: ✅ Camera ready');
    } catch (err) {
      DebugLog.error('Step 1: Camera FAILED', err);
      UILayer.hideLoading();
      const errMsg = document.createElement('div');
      errMsg.style.cssText = `
        position:fixed;inset:0;display:flex;align-items:center;justify-content:center;
        background:#000;color:rgba(255,255,255,0.5);font-size:14px;letter-spacing:0.15em;
        text-transform:uppercase;font-family:'JetBrains Mono',monospace;z-index:200;
      `;
      errMsg.textContent = 'Camera access required — please refresh and allow camera';
      document.body.appendChild(errMsg);
      return;
    }

    // ── Step 2: Init MediaPipe Hands ────────────────────────────────────
    DebugLog.log('Step 2: Initializing MediaPipe Hands...');
    try {
      TrackingEngine.init();
      DebugLog.log('Step 2: ✅ MediaPipe Hands initialized');
    } catch (err) {
      DebugLog.error('Step 2: MediaPipe init FAILED', err);
      // Non-fatal — we can still show the webcam without tracking
    }

    // ── Step 3: Hide loading screen and start render loop ──────────────
    DebugLog.log('Step 3: Hiding loading screen, starting render loop');
    UILayer.hideLoading();
    requestAnimationFrame(frame);
    DebugLog.log('Step 3: ✅ Render loop started');
  }

  function frame(timestamp) {
    requestAnimationFrame(frame);

    // Delta time
    if (!lastTimestamp) lastTimestamp = timestamp;
    const deltaMs = Math.min(timestamp - lastTimestamp, 50); // cap at 50ms
    const deltaS  = deltaMs / 1000;
    lastTimestamp  = timestamp;
    totalTime     += deltaS;
    frameCount++;

    // FPS
    const instantFPS = 1.0 / Math.max(deltaS, 0.001);
    fpsBuffer[fpsBufIdx] = instantFPS;
    fpsBufIdx = (fpsBufIdx + 1) % fpsBuffer.length;
    displayFPS = fpsBuffer.reduce((a, b) => a + b) / fpsBuffer.length;

    // Quality management
    const tier = QualityManager.update(displayFPS, deltaS);

    // ── Tracking ──────────────────────────────────────────────────────
    // Send current video frame to MediaPipe (non-blocking, skip if busy)
    const videoEl = WebcamSetup.videoEl;
    if (videoEl && videoEl.readyState >= 2 && !_mpBusy) {
      _mpBusy = true;
      TrackingEngine.send(videoEl).finally(() => { _mpBusy = false; });
    }

    const tracking = TrackingEngine.state;

    // Activation ramp
    const nowMs = performance.now();
    if (tracking.hasHand) {
      targetActivation = 1.0;
      UILayer.onHandDetected();
    } else {
      if (nowMs - tracking.lastSeenMs > CONFIG.LOST_TRACKING_MS) {
        targetActivation = 0.0;
      }
    }

    const rampInSpeed  = deltaMs / CONFIG.ACTIVATION_RAMP_IN_MS;
    const rampOutSpeed = deltaMs / CONFIG.ACTIVATION_RAMP_OUT_MS;
    const rampSpeed    = targetActivation > activationT ? rampInSpeed : rampOutSpeed;
    activationT += (targetActivation - activationT) * rampSpeed * 4.0;
    activationT  = Math.max(0, Math.min(1, activationT));

    // ── Gesture engine ─────────────────────────────────────────────────
    GestureEngine.update(tracking.rawLandmarks, deltaMs);
    const weights = GestureEngine.blendWeights;

    UILayer.updateGestureLabel(GestureEngine.stableGesture);

    // ── Smoothed uniforms ──────────────────────────────────────────────
    smoothVelocity += (tracking.velocity - smoothVelocity) * CONFIG.UNIFORM_SMOOTH;
    smoothAudio    += (AudioEngine.state.audioLevel - smoothAudio) * CONFIG.UNIFORM_SMOOTH;
    smoothDepth    += (tracking.depth - smoothDepth) * CONFIG.UNIFORM_SMOOTH;

    // Velocity delta for particle direction
    velDeltaX = tracking.fingerPos.x - prevFingerX;
    velDeltaY = tracking.fingerPos.y - prevFingerY;
    prevFingerX = tracking.fingerPos.x;
    prevFingerY = tracking.fingerPos.y;

    // Keep smoothed fingerPos + palmPos in WebGL engine uniforms
    WebGLEngine.uniforms.fingerPos[0] = tracking.fingerPos.x;
    WebGLEngine.uniforms.fingerPos[1] = tracking.fingerPos.y;
    WebGLEngine.uniforms.palmPos[0]   = tracking.palmPos.x;
    WebGLEngine.uniforms.palmPos[1]   = tracking.palmPos.y;

    // ── Audio engine ────────────────────────────────────────────────────
    AudioEngine.update();

    // ── WebGL render ────────────────────────────────────────────────────
    const currentVideo = WebcamSetup.videoEl;
    if (webglAvail && currentVideo) {
      WebGLEngine.render(
        currentVideo,
        totalTime,
        deltaS,
        glCanvas.width,
        glCanvas.height,
        weights,
        weights[G.IDLE],
        activationT,
        smoothVelocity,
        smoothAudio,
        smoothDepth,
        tier,
      );
    } else if (!webglAvail && currentVideo) {
      // Fallback 2D — just draw the webcam frame directly
      // (particles still render on overlay canvas)
    }

    // ── Overlay frame clear (once per frame) ──────────────────────────────
    ParticleSystem.beginFrame();

    // ── Particle system ─────────────────────────────────────────────────
    // Emit particles from all active hands
    const hs = TrackingEngine.handStates;
    if (CONFIG.TWO_HAND_MODE) {
      for (let i = 0; i < hs.length; i++) {
        if (hs[i].hasHand) {
          const fp = hs[i].fingerPos;
          const vdx = i === 0 ? velDeltaX : 0;
          const vdy = i === 0 ? velDeltaY : 0;
          ParticleSystem.update(
            deltaMs,
            fp,
            vdx, vdy,
            hs[i].velocity,
            activationT,
            true
          );
        }
      }
      // If no hands active, still call update to draw existing particles
      if (!hs[0].hasHand && !hs[1].hasHand) {
        ParticleSystem.update(deltaMs, tracking.fingerPos, 0, 0, 0, activationT, false);
      }
    } else {
      ParticleSystem.update(
        deltaMs,
        tracking.fingerPos,
        velDeltaX,
        velDeltaY,
        tracking.velocity,
        activationT,
        tracking.hasHand
      );
    }

    // ── Skeleton overlay ────────────────────────────────────────────────
    SkeletonRenderer.update(TrackingEngine.handStates, activationT);

    // ── Landmark data display ───────────────────────────────────────────
    LandmarkDisplay.update(TrackingEngine.handStates);

    // ── Debug overlay ───────────────────────────────────────────────────
    UILayer.updateDebug(
      displayFPS,
      GestureEngine.stableGesture,
      tracking.confidence,
      weights,
      ParticleSystem.activeCount,
      tier,
      smoothVelocity,
      smoothAudio,
    );
  }

  return { init };
})();

/* ════════════════════════════════════════════════════════════════════════════
   ENTRY POINT
   Wait for DOM ready, then start.
   ════════════════════════════════════════════════════════════════════════════ */
DebugLog.log('Entry point: readyState = ' + document.readyState);
if (document.readyState === 'loading') {
  DebugLog.log('Entry point: waiting for DOMContentLoaded...');
  document.addEventListener('DOMContentLoaded', () => {
    DebugLog.log('Entry point: DOMContentLoaded fired, calling init()');
    RenderLoop.init();
  });
} else {
  DebugLog.log('Entry point: DOM already ready, calling init() now');
  RenderLoop.init();
}
