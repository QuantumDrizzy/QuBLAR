// =============================================================================
// QuBLAR -- watertight ray/triangle intersection (Woop, Benthin & Wald, 2013)
// =============================================================================
// This replaces Möller–Trumbore in the traversal, and the reason is measured rather
// than aesthetic.
//
// Möller–Trumbore is not watertight. A ray striking a shared edge computes that edge
// from each of the two triangles using DIFFERENT expressions -- each triangle's own
// v0 is subtracted first -- so rounding can place the ray marginally outside both. The
// usual patch is to relax the barycentric bounds by an epsilon, which does not fix the
// class of error: it trades leaks for double hits and still leaks at some scale.
//
// The leak was real and it took a million rays to find. At subdiv 96 the RT-core
// comparison fired 1048576 incoherent rays into a closed box and the CUDA baseline
// missed exactly one that OptiX found. The existing watertightness test -- 30000 rays
// at subdiv 8 -- was three orders of magnitude too small to see it.
//
// Woop's construction makes the leak impossible rather than unlikely:
//
//   1. The ray is transformed into a space where it points down +z, by permuting axes
//      so the dominant component is last and shearing. The permutation depends only on
//      the RAY, so every triangle is measured in one common frame.
//   2. The edge functions U, V, W are 2D cross products of the sheared vertices. For two
//      triangles sharing an edge, the shared edge's function is computed from the SAME
//      two vertices in the same order, so the two results are exactly negatives of each
//      other. One of them is non-negative. Exactly one triangle claims the hit.
//   3. When an edge function lands exactly on zero, single precision cannot say which
//      side it fell on, so that one value is recomputed in double. This is the step that
//      turns "almost always watertight" into watertight, and it costs nothing in
//      practice because it fires only on exact zeros.
//
// The cost is a per-ray precomputation, which the traversal hoists out of its inner
// loop, and a few more operations per triangle than Möller–Trumbore. That is the right
// trade here: this simulator exists to study returns from edges, so a tracer that leaks
// at edges is the wrong baseline no matter how fast it is.
// =============================================================================

#pragma once

#include "ray.hpp"

/// Multiply and subtract WITHOUT letting the compiler fuse them.
///
/// This is not a micro-optimisation, it is the correctness of the whole scheme, and it
/// cost a confusing afternoon to find. Woop's guarantee is that two triangles sharing an
/// edge compute that edge's function to exactly opposite values. Written as `a*b - c*d`,
/// nvcc contracts the expression into an FMA, which rounds the product differently
/// depending on which of the two operands the multiply landed in -- so the two triangles
/// no longer produce exact negatives, both can round to the same side, and the ray leaks
/// between them.
///
/// The symptom was that the host build was perfect and the device build leaked, from
/// identical source: the host does not contract. `-fmad=false` fixes it and was rejected
/// as the fix, because it disables fusion across the entire program including the
/// radiometry, to solve a problem that lives in six expressions.
///
/// The exact-zero fallback below cannot cover for this: with contraction on, the value
/// is a tiny non-zero of arbitrary sign rather than a zero, so the fallback never fires.
#ifdef __CUDA_ARCH__
__device__ __forceinline__ float ex_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float ex_sub(float a, float b) { return __fsub_rn(a, b); }
#else
inline float ex_mul(float a, float b) { return a * b; }
inline float ex_sub(float a, float b) { return a - b; }
#endif

/// Per-ray constants: the axis permutation and the shear. Computed once per ray, never
/// per triangle -- it depends on the ray alone, which is exactly why the scheme is
/// consistent across triangles.
struct RayShear {
    int   kx, ky, kz;
    float Sx, Sy, Sz;
};

__host__ __device__ __forceinline__ float axis(const float3& v, int k) {
    return k == 0 ? v.x : (k == 1 ? v.y : v.z);
}

__host__ __device__ __forceinline__ RayShear make_shear(const float3& d) {
    RayShear s;
    const float ax = fabsf(d.x), ay = fabsf(d.y), az = fabsf(d.z);
    s.kz = (ax > ay) ? ((ax > az) ? 0 : 2) : ((ay > az) ? 1 : 2);
    s.kx = (s.kz + 1) % 3;
    s.ky = (s.kx + 1) % 3;

    // Swapping the two minor axes when the dominant component is negative preserves the
    // winding, so a triangle's edge functions do not flip sign with the ray's direction.
    if (axis(d, s.kz) < 0.0f) { const int t = s.kx; s.kx = s.ky; s.ky = t; }

    const float dz = axis(d, s.kz);
    s.Sz = 1.0f / dz;
    s.Sx = axis(d, s.kx) * s.Sz;
    s.Sy = axis(d, s.ky) * s.Sz;
    return s;
}

/// Nearest intersection, or false. `n_out` keeps the same convention the rest of the
/// code expects: cross(v1 - v0, v2 - v0), unnormalised, sign fixed by the caller.
__host__ __device__ __forceinline__ bool intersect_triangle_wt(
    const RayShear& s, const float3& o,
    const float3& v0, const float3& v1, const float3& v2,
    float tmax, float& t_out, float3& n_out)
{
    const float3 A = v0 - o;
    const float3 B = v1 - o;
    const float3 C = v2 - o;

    // The shear is unfused for the same reason as the edge functions: two triangles
    // sharing a vertex must shear it to bit-identical coordinates, or the edge functions
    // downstream are computed from different inputs and no amount of care in them helps.
    const float Ax = ex_sub(axis(A, s.kx), ex_mul(s.Sx, axis(A, s.kz)));
    const float Ay = ex_sub(axis(A, s.ky), ex_mul(s.Sy, axis(A, s.kz)));
    const float Bx = ex_sub(axis(B, s.kx), ex_mul(s.Sx, axis(B, s.kz)));
    const float By = ex_sub(axis(B, s.ky), ex_mul(s.Sy, axis(B, s.kz)));
    const float Cx = ex_sub(axis(C, s.kx), ex_mul(s.Sx, axis(C, s.kz)));
    const float Cy = ex_sub(axis(C, s.ky), ex_mul(s.Sy, axis(C, s.kz)));

    float U = ex_sub(ex_mul(Cx, By), ex_mul(Cy, Bx));
    float V = ex_sub(ex_mul(Ax, Cy), ex_mul(Ay, Cx));
    float W = ex_sub(ex_mul(Bx, Ay), ex_mul(By, Ax));

    // An exact zero means single precision has run out of information about which side
    // of the edge the ray passed. Redo that one in double, where the same expression is
    // exact for these operands, so the two triangles sharing the edge still disagree in
    // sign rather than both rounding to zero.
    if (U == 0.0f || V == 0.0f || W == 0.0f) {
        if (U == 0.0f) U = (float)((double)Cx * (double)By - (double)Cy * (double)Bx);
        if (V == 0.0f) V = (float)((double)Ax * (double)Cy - (double)Ay * (double)Cx);
        if (W == 0.0f) W = (float)((double)Bx * (double)Ay - (double)By * (double)Ax);
    }

    // Mixed signs mean the ray passes outside the triangle. Note this admits both
    // windings: back faces are hit too, which a closed box seen from the inside requires.
    if ((U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f))
        return false;

    const float det = U + V + W;
    if (det == 0.0f) return false;               // ray is edge-on to the plane

    const float Az = s.Sz * axis(A, s.kz);
    const float Bz = s.Sz * axis(B, s.kz);
    const float Cz = s.Sz * axis(C, s.kz);
    const float T  = U * Az + V * Bz + W * Cz;

    const float t = T / det;
    if (t <= 1e-6f || t >= tmax) return false;   // behind the origin, or past a known hit

    t_out = t;
    n_out = d_cross(v1 - v0, v2 - v0);
    return true;
}
