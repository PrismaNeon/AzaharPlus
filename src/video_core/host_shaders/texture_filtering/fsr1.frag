// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.
//
// AMD FidelityFX Super Resolution 1.0 adapted as a texture filter.
// EASU (Edge Adaptive Spatial Upsampling) followed by RCAS (Robust Contrast Adaptive Sharpening).
// Based on the algorithm in ffx_fsr1.h, Copyright (c) 2021 Advanced Micro Devices, Inc.,
// released under the MIT license.
//
// Texture filters run once when a texture is scaled, not every frame, so RCAS is done in the same
// pass by evaluating EASU at the centre pixel and at its four output-space neighbours.
// Colour is premultiplied by alpha while filtering so transparent texels do not bleed into the
// edges of sprites.

//? #version 430 core
precision highp float;

layout(location = 0) in vec2 tex_coord;
layout(location = 0) out vec4 frag_color;

layout(binding = 0) uniform sampler2D tex;

#ifdef VULKAN
layout(push_constant) uniform PushConstants {
    layout(offset = 16) float res_scale;
};
#else
layout(location = 2) uniform float res_scale;
#endif

// RCAS strength in stops: 0.0 is the strongest sharpening, each +1.0 halves it.
const float RCAS_SHARPNESS_STOPS = 0.25;
const float RCAS_LIMIT = 0.25 - (1.0 / 16.0);

vec2 src_size;

vec4 Fetch(vec2 texel) {
    ivec2 p = clamp(ivec2(texel), ivec2(0), ivec2(src_size) - ivec2(1));
    vec4 c = texelFetch(tex, p, 0);
    return vec4(c.rgb * c.a, c.a);
}

float Luma(vec4 c) {
    return c.b * 0.5 + (c.r * 0.5 + c.g);
}

void EasuSet(inout vec2 dir, inout float len, float w, float lA, float lB, float lC, float lD,
             float lE) {
    // Horizontal gradient
    float lenX = max(abs(lD - lC), abs(lC - lB));
    lenX = 1.0 / max(lenX, 1.0e-5);
    float dirX = lD - lB;
    dir.x += dirX * w;
    lenX = clamp(abs(dirX) * lenX, 0.0, 1.0);
    lenX *= lenX;
    len += lenX * w;
    // Vertical gradient
    float lenY = max(abs(lE - lC), abs(lC - lA));
    lenY = 1.0 / max(lenY, 1.0e-5);
    float dirY = lE - lA;
    dir.y += dirY * w;
    lenY = clamp(abs(dirY) * lenY, 0.0, 1.0);
    lenY *= lenY;
    len += lenY * w;
}

void EasuTap(inout vec4 aC, inout float aW, vec2 off, vec2 dir, vec2 len2, float lob, float clp,
             vec4 c) {
    // Rotate the offset into the edge direction and apply anisotropic stretch
    vec2 v = vec2(dot(off, dir), dot(off, vec2(-dir.y, dir.x))) * len2;
    float d2 = min(dot(v, v), clp);
    // Approximation of lanczos2 without sin() or rcp()
    float wB = 0.4 * d2 - 1.0;
    float wA = lob * d2 - 1.0;
    wB *= wB;
    wA *= wA;
    wB = 1.5625 * wB - 0.5625;
    float w = wB * wA;
    aC += c * w;
    aW += w;
}

vec4 Easu(vec2 uv) {
    vec2 pp = uv * src_size - vec2(0.5);
    vec2 fp = floor(pp);
    pp -= fp;

    // 12-tap kernel:
    //    b c
    //  e f g h
    //  i j k l
    //    n o
    vec4 b = Fetch(fp + vec2(0.0, -1.0));
    vec4 c = Fetch(fp + vec2(1.0, -1.0));
    vec4 e = Fetch(fp + vec2(-1.0, 0.0));
    vec4 f = Fetch(fp + vec2(0.0, 0.0));
    vec4 g = Fetch(fp + vec2(1.0, 0.0));
    vec4 h = Fetch(fp + vec2(2.0, 0.0));
    vec4 i = Fetch(fp + vec2(-1.0, 1.0));
    vec4 j = Fetch(fp + vec2(0.0, 1.0));
    vec4 k = Fetch(fp + vec2(1.0, 1.0));
    vec4 l = Fetch(fp + vec2(2.0, 1.0));
    vec4 n = Fetch(fp + vec2(0.0, 2.0));
    vec4 o = Fetch(fp + vec2(1.0, 2.0));

    float bL = Luma(b);
    float cL = Luma(c);
    float eL = Luma(e);
    float fL = Luma(f);
    float gL = Luma(g);
    float hL = Luma(h);
    float iL = Luma(i);
    float jL = Luma(j);
    float kL = Luma(k);
    float lL = Luma(l);
    float nL = Luma(n);
    float oL = Luma(o);

    // Accumulate edge direction and length from the four bilinear neighbourhoods
    vec2 dir = vec2(0.0);
    float len = 0.0;
    EasuSet(dir, len, (1.0 - pp.x) * (1.0 - pp.y), bL, eL, fL, gL, jL);
    EasuSet(dir, len, pp.x * (1.0 - pp.y), cL, fL, gL, hL, kL);
    EasuSet(dir, len, (1.0 - pp.x) * pp.y, fL, iL, jL, kL, nL);
    EasuSet(dir, len, pp.x * pp.y, gL, jL, kL, lL, oL);

    // Normalize direction, falling back to horizontal when there is no gradient
    vec2 dir2 = dir * dir;
    float dirR = dir2.x + dir2.y;
    bool zro = dirR < (1.0 / 32768.0);
    dirR = zro ? 1.0 : inversesqrt(dirR);
    dir.x = zro ? 1.0 : dir.x;
    dir *= dirR;

    // Shape the kernel: stretch along the edge, shrink across it
    len = len * 0.5;
    len *= len;
    float stretch = dot(dir, dir) / max(abs(dir.x), abs(dir.y));
    vec2 len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    float lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
    float clp = 1.0 / lob;

    vec4 aC = vec4(0.0);
    float aW = 0.0;
    EasuTap(aC, aW, vec2(0.0, -1.0) - pp, dir, len2, lob, clp, b);
    EasuTap(aC, aW, vec2(1.0, -1.0) - pp, dir, len2, lob, clp, c);
    EasuTap(aC, aW, vec2(-1.0, 1.0) - pp, dir, len2, lob, clp, i);
    EasuTap(aC, aW, vec2(0.0, 1.0) - pp, dir, len2, lob, clp, j);
    EasuTap(aC, aW, vec2(0.0, 0.0) - pp, dir, len2, lob, clp, f);
    EasuTap(aC, aW, vec2(-1.0, 0.0) - pp, dir, len2, lob, clp, e);
    EasuTap(aC, aW, vec2(1.0, 1.0) - pp, dir, len2, lob, clp, k);
    EasuTap(aC, aW, vec2(2.0, 1.0) - pp, dir, len2, lob, clp, l);
    EasuTap(aC, aW, vec2(2.0, 0.0) - pp, dir, len2, lob, clp, h);
    EasuTap(aC, aW, vec2(1.0, 0.0) - pp, dir, len2, lob, clp, g);
    EasuTap(aC, aW, vec2(1.0, 2.0) - pp, dir, len2, lob, clp, o);
    EasuTap(aC, aW, vec2(0.0, 2.0) - pp, dir, len2, lob, clp, n);

    vec4 pix = abs(aW) > 1.0e-5 ? aC / aW : f;

    // Deringing: clamp to the range of the nearest 2x2 texels
    vec4 mn = min(min(f, g), min(j, k));
    vec4 mx = max(max(f, g), max(j, k));
    return clamp(pix, mn, mx);
}

void main() {
    src_size = vec2(textureSize(tex, 0));
    vec2 px = 1.0 / (src_size * max(res_scale, 1.0));

    // EASU at the output pixel and its four neighbours (cross pattern for RCAS)
    //    b
    //  d e f
    //    h
    vec4 e = Easu(tex_coord);
    vec4 b = Easu(tex_coord + vec2(0.0, -px.y));
    vec4 d = Easu(tex_coord + vec2(-px.x, 0.0));
    vec4 f = Easu(tex_coord + vec2(px.x, 0.0));
    vec4 h = Easu(tex_coord + vec2(0.0, px.y));

    // RCAS: solve for the maximum sharpening lobe that does not clip
    vec3 mn4 = min(min(b.rgb, d.rgb), min(f.rgb, h.rgb));
    vec3 mx4 = max(max(b.rgb, d.rgb), max(f.rgb, h.rgb));
    vec3 hitMin = min(mn4, e.rgb) / max(4.0 * mx4, vec3(1.0e-5));
    vec3 hitMax = (1.0 - max(mx4, e.rgb)) / min(4.0 * mn4 - 4.0, vec3(-1.0e-5));
    vec3 lobeRGB = max(-hitMin, hitMax);
    float lobe = max(-RCAS_LIMIT, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) *
                 exp2(-RCAS_SHARPNESS_STOPS);
    float rcpL = 1.0 / (4.0 * lobe + 1.0);

    vec4 result;
    result.rgb = (lobe * (b.rgb + d.rgb + h.rgb + f.rgb) + e.rgb) * rcpL;
    result.a = e.a;

    // Back to straight alpha
    result.rgb = clamp(result.rgb, vec3(0.0), vec3(result.a));
    result.rgb = result.a > 0.0 ? result.rgb / result.a : vec3(0.0);
    frag_color = result;
}
