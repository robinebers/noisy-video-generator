#!/usr/bin/env python3
"""
noisy-video-generator

GPU-accelerated procedural video generator. Renders domain-warped fBM
simplex noise via GLSL shaders (moderngl headless), applies multi-pass
Gaussian blur, and pipes frames to ffmpeg for H.264 MP4 output.

Usage:
    python generate.py --primary "#1535C4" --secondary "#C8B8FF" --output out.mp4
"""

import argparse
import colorsys
import subprocess
import sys

import moderngl
import numpy as np

# ---------------------------------------------------------------------------
# GLSL sources
# ---------------------------------------------------------------------------

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

NOISE_FRAGMENT = """
#version 330
precision highp float;

in vec2 v_uv;
out vec4 frag_color;

uniform float u_time;
uniform vec2  u_resolution;
uniform float u_speed;
uniform float u_noise_scale;
uniform int   u_octaves;
uniform float u_warp;
uniform float u_grain;
uniform float u_seed;

uniform vec3 u_col_deep;
uniform vec3 u_col_primary;
uniform vec3 u_col_mid;
uniform vec3 u_col_secondary;
uniform vec3 u_col_highlight;
uniform vec3 u_col_accent;

// --- Simplex 4D noise (Stefan Gustavson, public domain) ---

vec4 mod289(vec4 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
float mod289(float x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+10.0)*x); }
float permute(float x) { return mod289(((x*34.0)+10.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
float taylorInvSqrt(float r) { return 1.79284291400159 - 0.85373472095314 * r; }

vec4 grad4(float j, vec4 ip) {
    const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
    vec4 p, s;
    p.xyz = floor(fract(vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
    p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
    s = vec4(lessThan(p, vec4(0.0)));
    p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www;
    return p;
}

#define F4 0.309016994374947451

float snoise(vec4 v) {
    const vec4 C = vec4(
        0.138196601125011,
        0.276393202250021,
        0.414589803375032,
       -0.447213595499958);

    vec4 i  = floor(v + dot(v, vec4(F4)));
    vec4 x0 = v - i + dot(i, C.xxxx);

    vec4 i0;
    vec3 isX = step(x0.yzw, x0.xxx);
    vec3 isYZ = step(x0.zww, x0.yyz);
    i0.x = isX.x + isX.y + isX.z;
    i0.yzw = 1.0 - isX;
    i0.y += isYZ.x + isYZ.y;
    i0.zw += 1.0 - isYZ.xy;
    i0.z += isYZ.z;
    i0.w += 1.0 - isYZ.z;

    vec4 i3 = clamp(i0, 0.0, 1.0);
    vec4 i2 = clamp(i0 - 1.0, 0.0, 1.0);
    vec4 i1 = clamp(i0 - 2.0, 0.0, 1.0);

    vec4 x1 = x0 - i1 + C.xxxx;
    vec4 x2 = x0 - i2 + C.yyyy;
    vec4 x3 = x0 - i3 + C.zzzz;
    vec4 x4 = x0 + C.wwww;

    i = mod289(i);
    float j0 = permute(permute(permute(permute(i.w)+i.z)+i.y)+i.x);
    vec4 j1 = permute(permute(permute(permute(
        i.w + vec4(i1.w, i2.w, i3.w, 1.0))
      + i.z + vec4(i1.z, i2.z, i3.z, 1.0))
      + i.y + vec4(i1.y, i2.y, i3.y, 1.0))
      + i.x + vec4(i1.x, i2.x, i3.x, 1.0));

    vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0);

    vec4 p0 = grad4(j0,   ip);
    vec4 p1 = grad4(j1.x, ip);
    vec4 p2 = grad4(j1.y, ip);
    vec4 p3 = grad4(j1.z, ip);
    vec4 p4 = grad4(j1.w, ip);

    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    p4 *= taylorInvSqrt(dot(p4,p4));

    vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
    vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)), 0.0);
    m0 = m0 * m0; m1 = m1 * m1;
    return 49.0 * (
        dot(m0*m0, vec3(dot(p0,x0), dot(p1,x1), dot(p2,x2)))
      + dot(m1*m1, vec2(dot(p3,x3), dot(p4,x4)))
    );
}

float fbm4(vec4 p, int octs) {
    float val = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 8; i++) {
        if (i >= octs) break;
        val += amp * snoise(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return val;
}

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    vec2 uv = v_uv;
    float aspect = u_resolution.x / u_resolution.y;
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0) * u_noise_scale;

    float angle = u_time * 6.2831853;
    float loop_r = 0.6 * u_speed;
    float tz = cos(angle) * loop_r;
    float tw = sin(angle) * loop_r;

    vec2 seed_off = vec2(u_seed * 13.37, u_seed * 7.13);
    p += seed_off;

    // Domain warping pass 1
    float warp1_x = fbm4(vec4(p + vec2(0.0, 0.0), tz, tw), u_octaves);
    float warp1_y = fbm4(vec4(p + vec2(5.2, 1.3), tz, tw), u_octaves);
    vec2 warped1 = p + u_warp * vec2(warp1_x, warp1_y);

    // Domain warping pass 2 (warp-on-warp)
    float warp2_x = fbm4(vec4(warped1 + vec2(1.7, 9.2), tz, tw), u_octaves);
    float warp2_y = fbm4(vec4(warped1 + vec2(8.3, 2.8), tz, tw), u_octaves);
    vec2 warped2 = p + u_warp * vec2(warp2_x, warp2_y);

    float n = fbm4(vec4(warped2, tz, tw), u_octaves);
    n = n * 0.5 + 0.5;
    n = smoothstep(0.1, 0.9, n);

    // Color palette mapping
    vec3 col;
    if (n < 0.25) {
        col = mix(u_col_deep, u_col_primary, n / 0.25);
    } else if (n < 0.5) {
        col = mix(u_col_primary, u_col_mid, (n - 0.25) / 0.25);
    } else if (n < 0.75) {
        col = mix(u_col_mid, u_col_secondary, (n - 0.5) / 0.25);
    } else {
        col = mix(u_col_secondary, u_col_highlight, (n - 0.75) / 0.25);
    }

    float accent_mask = smoothstep(0.3, 0.5, n) * smoothstep(0.7, 0.5, n);
    col = mix(col, u_col_accent, accent_mask * 0.15);

    // Grain (applied before blur so it gets softened too)
    float grain = (hash(uv * u_resolution + u_time * 1000.0) - 0.5) * u_grain;
    col += grain;

    col = clamp(col, 0.0, 1.0);
    frag_color = vec4(col, 1.0);
}
"""

BLUR_FRAGMENT = """
#version 330
precision highp float;

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_tex;
uniform vec2 u_direction;
uniform float u_sigma;

void main() {
    vec4 color = vec4(0.0);
    float total = 0.0;
    float sigma2 = u_sigma * u_sigma;
    int radius = int(ceil(u_sigma * 3.0));

    for (int i = -48; i <= 48; i++) {
        if (i < -radius || i > radius) continue;
        float w = exp(-float(i * i) / (2.0 * sigma2));
        color += texture(u_tex, v_uv + float(i) * u_direction) * w;
        total += w;
    }
    frag_color = color / total;
}
"""

# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hsl(r: float, g: float, b: float):
    return colorsys.rgb_to_hls(r, g, b)


def hsl_to_rgb(h: float, l: float, s: float):
    return colorsys.hls_to_rgb(h, l, s)


def darken(rgb, amount=0.2):
    h, l, s = rgb_to_hsl(*rgb)
    return hsl_to_rgb(h, max(0.0, l - amount), s)


def lighten(rgb, amount=0.2):
    h, l, s = rgb_to_hsl(*rgb)
    return hsl_to_rgb(h, min(1.0, l + amount), s)


def hue_shift(rgb, degrees=15):
    h, l, s = rgb_to_hsl(*rgb)
    h = (h + degrees / 360.0) % 1.0
    return hsl_to_rgb(h, l, s)


def lerp_color(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def infer_palette(primary_hex: str, secondary_hex: str):
    primary = hex_to_rgb(primary_hex)
    secondary = hex_to_rgb(secondary_hex)
    return {
        "deep": darken(primary, 0.15),
        "primary": primary,
        "mid": lerp_color(primary, secondary, 0.5),
        "secondary": secondary,
        "highlight": lighten(secondary, 0.2),
        "accent": hue_shift(secondary, 15),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def build_pipeline(ctx, width, height):
    """Create noise program, blur program, quad VAO, and FBOs."""
    noise_prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=NOISE_FRAGMENT)
    blur_prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=BLUR_FRAGMENT)

    vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
    vbo = ctx.buffer(vertices)
    noise_vao = ctx.simple_vertex_array(noise_prog, vbo, "in_pos")
    blur_vao = ctx.simple_vertex_array(blur_prog, vbo, "in_pos")

    # Noise renders to tex_a
    tex_a = ctx.texture((width, height), 3)
    tex_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
    fbo_a = ctx.framebuffer(color_attachments=[tex_a])

    # Horizontal blur reads tex_a, writes to tex_b
    tex_b = ctx.texture((width, height), 3)
    tex_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
    fbo_b = ctx.framebuffer(color_attachments=[tex_b])

    # Vertical blur reads tex_b, writes to tex_c (final output)
    tex_c = ctx.texture((width, height), 3)
    tex_c.filter = (moderngl.LINEAR, moderngl.LINEAR)
    fbo_c = ctx.framebuffer(color_attachments=[tex_c])

    return {
        "noise_prog": noise_prog,
        "blur_prog": blur_prog,
        "noise_vao": noise_vao,
        "blur_vao": blur_vao,
        "tex_a": tex_a, "fbo_a": fbo_a,
        "tex_b": tex_b, "fbo_b": fbo_b,
        "tex_c": tex_c, "fbo_c": fbo_c,
    }


def start_ffmpeg(width: int, height: int, fps: int, output: str):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        "-movflags", "+faststart",
        output,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def render_video(args):
    width, height = args.width, args.height
    total_frames = int(args.duration * args.fps)
    blur_sigma = args.blur

    palette = infer_palette(args.primary, args.secondary)

    ctx = moderngl.create_standalone_context()
    p = build_pipeline(ctx, width, height)

    # Set noise uniforms
    np_ = p["noise_prog"]
    np_["u_resolution"].value = (float(width), float(height))
    np_["u_speed"].value = args.speed
    np_["u_noise_scale"].value = args.noise_scale
    np_["u_octaves"].value = args.octaves
    np_["u_warp"].value = args.warp
    np_["u_grain"].value = args.grain
    np_["u_seed"].value = float(args.seed)
    for name, rgb in palette.items():
        np_[f"u_col_{name}"].value = rgb

    # Set blur uniforms (static)
    bp = p["blur_prog"]
    bp["u_tex"].value = 0
    bp["u_sigma"].value = blur_sigma

    pixel_w = 1.0 / width
    pixel_h = 1.0 / height

    ffmpeg = start_ffmpeg(width, height, args.fps, args.output)

    print(f"Rendering {total_frames} frames ({args.duration}s @ {args.fps}fps) "
          f"at {width}x{height}, blur sigma={blur_sigma}")

    for i in range(total_frames):
        t = i / total_frames
        np_["u_time"].value = t

        # Pass 1: render noise to fbo_a
        p["fbo_a"].use()
        p["noise_vao"].render(moderngl.TRIANGLE_STRIP)

        if blur_sigma > 0:
            # Pass 2: horizontal blur (tex_a → fbo_b)
            bp["u_direction"].value = (pixel_w, 0.0)
            p["tex_a"].use(location=0)
            p["fbo_b"].use()
            p["blur_vao"].render(moderngl.TRIANGLE_STRIP)

            # Pass 3: vertical blur (tex_b → fbo_c)
            bp["u_direction"].value = (0.0, pixel_h)
            p["tex_b"].use(location=0)
            p["fbo_c"].use()
            p["blur_vao"].render(moderngl.TRIANGLE_STRIP)

            raw = p["fbo_c"].read(components=3)
        else:
            raw = p["fbo_a"].read(components=3)

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        frame = np.flipud(frame)
        ffmpeg.stdin.write(frame.tobytes())

        if (i + 1) % 30 == 0 or i == total_frames - 1:
            pct = (i + 1) / total_frames * 100
            print(f"  frame {i+1}/{total_frames} ({pct:.0f}%)")

    ffmpeg.stdin.close()
    ffmpeg.wait()

    if ffmpeg.returncode != 0:
        err = ffmpeg.stderr.read().decode()
        print(f"ffmpeg error:\n{err}", file=sys.stderr)
        sys.exit(1)

    print(f"Done -> {args.output}")
    ctx.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate procedural background video")
    ap.add_argument("--primary", default="#2b7fff", help="Primary color hex (default: #2b7fff)")
    ap.add_argument("--secondary", default="#b8d8ff", help="Secondary color hex (default: #b8d8ff)")
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--duration", type=float, default=6, help="Seconds (default: 6)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=0.15, help="Animation speed (default: 0.15)")
    ap.add_argument("--noise-scale", type=float, default=0.7,
                    help="Noise feature scale; lower = larger shapes (default: 0.7)")
    ap.add_argument("--octaves", type=int, default=2,
                    help="fBM octaves; fewer = smoother (default: 2)")
    ap.add_argument("--warp", type=float, default=1.5,
                    help="Domain warp strength (default: 1.5)")
    ap.add_argument("--grain", type=float, default=0.03,
                    help="Grain intensity 0-1 (default: 0.03)")
    ap.add_argument("--blur", type=float, default=20.0,
                    help="Gaussian blur sigma in px; 0 = off (default: 20.0)")
    ap.add_argument("--seed", type=float, default=42)
    ap.add_argument("--output", default="output.mp4")

    render_video(ap.parse_args())


if __name__ == "__main__":
    main()
