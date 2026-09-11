// =============================================================================
// QuBLAR -- ray/scene intersection on the GPU
// =============================================================================
// The traversal, the intersector, and the procedural scenes whose answers are known
// in closed form. Phase 1 of ADR-001, and the baseline the OptiX path is measured
// against, so nothing here may be "close enough": the two implementations have to
// agree hit for hit before a speedup number is allowed to mean anything.
//
// A header rather than a translation unit because two binaries need it -- the
// correctness harness (trace.cu) and the LiDAR sensor model (check_lidar.cu) -- and
// the one thing that must not happen is the physics tracing a second, slightly
// different copy of the geometry.
// =============================================================================

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "bvh.hpp"

using namespace argos;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t _e = (call);                                                      \
        if (_e != cudaSuccess) {                                                      \
            std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(_e),        \
                         __FILE__, __LINE__);                                         \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

#include "ray.hpp"
#include "watertight.hpp"

/// Moeller-Trumbore lived here and was removed, not merely superseded.
///
/// It is faster and it is not watertight: the two triangles sharing an edge evaluate
/// that edge from different expressions, so a grazing ray can round outside both. The
/// barycentric epsilon that used to sit here made the leak rarer without making it
/// impossible -- and a rare wrong answer in a baseline is worse than a slow one, because
/// everything measured against it inherits the error silently. See watertight.hpp.

/// Reciprocal of a direction, with zero components nudged so the slab test cannot
/// produce a NaN.
///
/// This is what the six watertightness leaks actually were, and the first diagnosis was
/// wrong: an axis-aligned ray has a zero direction component, so 1/0 is +inf, and a box
/// face lying exactly on that axis gives (lo - o) * inf = 0 * inf = NaN. The comparison
/// against NaN is false, the node is rejected, and the ray sails through solid geometry.
///
/// It is invisible until a ray is fired exactly down an axis at a box whose split planes
/// pass through the origin -- which is precisely what the test does, and precisely what a
/// scanning LiDAR does every time its beam is level with the sensor.
///
/// Nudging to 1e-20 keeps the reciprocal finite, so the product is 0 rather than NaN and
/// the slab test does what it means to.
__host__ __device__ __forceinline__ float3 safe_inverse(const float3& d) {
    const float ex = fabsf(d.x) < 1e-20f ? copysignf(1e-20f, d.x) : d.x;
    const float ey = fabsf(d.y) < 1e-20f ? copysignf(1e-20f, d.y) : d.y;
    const float ez = fabsf(d.z) < 1e-20f ? copysignf(1e-20f, d.z) : d.z;
    return make_float3(1.0f / ex, 1.0f / ey, 1.0f / ez);
}

/// Slab test. Returns the near distance, or -1 when the ray misses the box.
///
/// Written with the reciprocal precomputed by the caller so the divide is not paid per
/// node -- traversal visits many nodes per ray and this is the inner loop.
__host__ __device__ __forceinline__ float intersect_aabb(
    const float3& o, const float3& inv_d, const float3& lo, const float3& hi, float tmax)
{
    const float tx1 = (lo.x - o.x) * inv_d.x, tx2 = (hi.x - o.x) * inv_d.x;
    float tmin = fminf(tx1, tx2), tmaxv = fmaxf(tx1, tx2);
    const float ty1 = (lo.y - o.y) * inv_d.y, ty2 = (hi.y - o.y) * inv_d.y;
    tmin = fmaxf(tmin, fminf(ty1, ty2)); tmaxv = fminf(tmaxv, fmaxf(ty1, ty2));
    const float tz1 = (lo.z - o.z) * inv_d.z, tz2 = (hi.z - o.z) * inv_d.z;
    tmin = fmaxf(tmin, fminf(tz1, tz2)); tmaxv = fminf(tmaxv, fmaxf(tz1, tz2));

    // The <= is load-bearing: a ray lying exactly in the plane of a face produces
    // tmin == tmaxv, and rejecting that would put a seam through every axis-aligned wall.
    return (tmaxv >= fmaxf(tmin, 0.0f) && tmin < tmax) ? fmaxf(tmin, 0.0f) : -1.0f;
}

// -----------------------------------------------------------------------------

constexpr int kStackSize = 32;   // 2^32 leaves is far past what any scene here will hold

/// Nearest hit for one ray, walking the hierarchy with an explicit stack.
///
/// A device function rather than a kernel body because the LiDAR layer calls it many
/// times per beam -- once per footprint sample -- and duplicating a traversal is how the
/// baseline and the physics quietly stop tracing the same geometry.
__host__ __device__ __forceinline__ Hit traverse_bvh(
    const BvhNode* __restrict__ nodes,
    const int* __restrict__ indices,
    const Triangle* __restrict__ tris,
    const Ray& r)
{
    const float3 inv_d = safe_inverse(r.direction);
    const RayShear shear = make_shear(r.direction);   // per ray, never per triangle

    float best_t = r.tmax;
    int   best_tri = -1;
    float3 best_n = make_float3(0.f, 0.f, 0.f);

    int stack[kStackSize];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const int ni = stack[--sp];
        const BvhNode n = nodes[ni];

        const float3 lo = make_float3(n.lo.x, n.lo.y, n.lo.z);
        const float3 hi = make_float3(n.hi.x, n.hi.y, n.hi.z);
        if (intersect_aabb(r.origin, inv_d, lo, hi, best_t) < 0.0f) continue;

        if (n.count > 0) {
            for (int k = 0; k < n.count; ++k) {
                const int ti = indices[n.left_first + k];
                const Triangle tr = tris[ti];
                const float3 v0 = make_float3(tr.v0.x, tr.v0.y, tr.v0.z);
                const float3 v1 = make_float3(tr.v1.x, tr.v1.y, tr.v1.z);
                const float3 v2 = make_float3(tr.v2.x, tr.v2.y, tr.v2.z);
                float t; float3 nrm;
                if (intersect_triangle_wt(shear, r.origin, v0, v1, v2, best_t, t, nrm)) {
                    best_t = t; best_tri = ti; best_n = nrm;
                }
            }
        } else {
            // Push both children. Ordering them near-first would cut node visits, and is
            // deliberately left out of the baseline -- see the header.
            if (sp + 2 <= kStackSize) {
                stack[sp++] = n.left_first;   // right child
                stack[sp++] = ni + 1;         // left child, visited first
            }
        }
    }

    Hit h;
    h.t = (best_tri >= 0) ? best_t : -1.0f;
    h.triangle = best_tri;
    h.material = (best_tri >= 0) ? tris[best_tri].material : -1;
    if (best_tri >= 0) {
        // Face the normal against the ray so cos(theta) in the radiometry is unsigned.
        const float len = sqrtf(d_dot(best_n, best_n));
        float3 nn = (len > 0.f) ? best_n * (1.0f / len) : best_n;
        if (d_dot(nn, r.direction) > 0.f) nn = nn * -1.0f;
        h.normal = nn;
    } else {
        h.normal = make_float3(0.f, 0.f, 0.f);
    }
    return h;
}

/// Every ray against every triangle. The oracle: no hierarchy, nothing to get wrong
/// except the intersector itself, which the analytic test covers separately.
__host__ __device__ __forceinline__ Hit traverse_brute(
    const Triangle* __restrict__ tris, int n_tris, const Ray& r)
{
    const RayShear shear = make_shear(r.direction);
    float best_t = r.tmax;
    int best_tri = -1;
    float3 best_n = make_float3(0.f, 0.f, 0.f);

    for (int ti = 0; ti < n_tris; ++ti) {
        const Triangle tr = tris[ti];
        const float3 v0 = make_float3(tr.v0.x, tr.v0.y, tr.v0.z);
        const float3 v1 = make_float3(tr.v1.x, tr.v1.y, tr.v1.z);
        const float3 v2 = make_float3(tr.v2.x, tr.v2.y, tr.v2.z);
        float t; float3 nrm;
        if (intersect_triangle_wt(shear, r.origin, v0, v1, v2, best_t, t, nrm)) {
            best_t = t; best_tri = ti; best_n = nrm;
        }
    }

    Hit h;
    h.t = (best_tri >= 0) ? best_t : -1.0f;
    h.triangle = best_tri;
    h.material = (best_tri >= 0) ? tris[best_tri].material : -1;
    if (best_tri >= 0) {
        const float len = sqrtf(d_dot(best_n, best_n));
        float3 nn = (len > 0.f) ? best_n * (1.0f / len) : best_n;
        if (d_dot(nn, r.direction) > 0.f) nn = nn * -1.0f;
        h.normal = nn;
    } else {
        h.normal = make_float3(0.f, 0.f, 0.f);
    }
    return h;
}


// -----------------------------------------------------------------------------
// Kernel wrappers. Thin on purpose: the traversal above is the thing under test, and
// the LiDAR layer must exercise exactly it and not a near-copy.
// -----------------------------------------------------------------------------
__global__ void trace_bvh(
    const BvhNode* __restrict__ nodes,
    const int* __restrict__ indices,
    const Triangle* __restrict__ tris,
    const Ray* __restrict__ rays,
    Hit* __restrict__ hits,
    int n_rays)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rays) hits[i] = traverse_bvh(nodes, indices, tris, rays[i]);
}

__global__ void trace_brute(
    const Triangle* __restrict__ tris, int n_tris,
    const Ray* __restrict__ rays, Hit* __restrict__ hits, int n_rays)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rays) hits[i] = traverse_brute(tris, n_tris, rays[i]);
}

// -----------------------------------------------------------------------------
// Scenes. Procedural, because validation needs geometry whose answer is known
// exactly, and a scanned mesh does not give you that (ADR-001, open question).
// -----------------------------------------------------------------------------

/// A plane at range `d` along +z, tilted `tilt_deg` about the y axis.
/// Every ray fired along +z from the origin has an analytic hit distance.
static Scene make_tilted_plane(float d, float tilt_deg, float half = 20.f) {
    Scene s;
    s.add_material(0.4f);
    const float a = tilt_deg * 3.14159265358979f / 180.f;
    const float ca = std::cos(a), sa = std::sin(a);
    // Rotate the plane's corners about y. Points satisfy z = d + x*tan(a) after rotation.
    auto pt = [&](float x, float y) {
        return Vec3{x * ca, y, d + x * sa};
    };
    s.add_quad(pt(-half, -half), pt(half, -half), pt(half, half), pt(-half, half), 0);
    s.build();
    return s;
}

/// A closed box, which is where a leaky intersector shows itself: interior rays that
/// graze a corner must hit something, never escape.
static Scene make_box(float size, int subdiv) {
    Scene s;
    s.add_material(0.5f);
    const float h = size * 0.5f;
    const float step = size / subdiv;
    // Subdivided so shared edges are numerous -- the watertightness test needs them.
    for (int i = 0; i < subdiv; ++i) {
        for (int j = 0; j < subdiv; ++j) {
            // Both edges of a cell are computed from the SAME expression, and this is
            // not pedantry. Writing x1 = x0 + step makes cell i's right edge a different
            // float from cell i+1's left edge -- they differ by about one ULP, roughly
            // 2.4e-7 at this magnitude, and that gap is a real crack between two quads
            // that are supposed to share an edge.
            //
            // It stayed invisible for a long time. At subdiv 6 with 30000 rays the cracks
            // are too coarse and too few to be struck. It surfaced only when the RT-core
            // comparison fired a million incoherent rays at a subdiv-96 box: three of them
            // escaped a closed box, two missed by OptiX and one by the CUDA baseline. That
            // split is the whole diagnosis -- OptiX's intersector is watertight by
            // construction, so a leak that hits BOTH tracers cannot be in either of them.
            // The scene was open, and the generator opened it.
            const float x0 = -h + i * step, x1 = -h + (i + 1) * step;
            const float y0 = -h + j * step, y1 = -h + (j + 1) * step;
            s.add_quad({x0, y0, h}, {x1, y0, h}, {x1, y1, h}, {x0, y1, h}, 0);   // +z
            s.add_quad({x0, y1, -h}, {x1, y1, -h}, {x1, y0, -h}, {x0, y0, -h}, 0);
            s.add_quad({h, y0, x0}, {h, y1, x0}, {h, y1, x1}, {h, y0, x1}, 0);   // +x
            s.add_quad({-h, y0, x1}, {-h, y1, x1}, {-h, y1, x0}, {-h, y0, x0}, 0);
            s.add_quad({x0, h, y0}, {x1, h, y0}, {x1, h, y1}, {x0, h, y1}, 0);   // +y
            s.add_quad({x0, -h, y1}, {x1, -h, y1}, {x1, -h, y0}, {x0, -h, y0}, 0);
        }
    }
    s.build();
    return s;
}

// -----------------------------------------------------------------------------

struct DeviceScene {
    BvhNode*  nodes = nullptr;
    int*      indices = nullptr;
    Triangle* tris = nullptr;
    Material* mats = nullptr;
    int n_tris = 0;

    void upload(const Scene& s) {
        n_tris = static_cast<int>(s.triangles.size());
        CUDA_CHECK(cudaMalloc(&mats, s.materials.size() * sizeof(Material)));
        CUDA_CHECK(cudaMemcpy(mats, s.materials.data(), s.materials.size() * sizeof(Material),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&nodes, s.nodes.size() * sizeof(BvhNode)));
        CUDA_CHECK(cudaMalloc(&indices, s.indices.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&tris, s.triangles.size() * sizeof(Triangle)));
        CUDA_CHECK(cudaMemcpy(nodes, s.nodes.data(), s.nodes.size() * sizeof(BvhNode),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(indices, s.indices.data(), s.indices.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tris, s.triangles.data(), s.triangles.size() * sizeof(Triangle),
                              cudaMemcpyHostToDevice));
    }
    void free() {
        cudaFree(nodes); cudaFree(indices); cudaFree(tris); cudaFree(mats);
    }
};

