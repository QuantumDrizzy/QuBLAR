// =============================================================================
// ARGOS -- CUDA BVH traversal, and the correctness harness for it
// =============================================================================
// Rays in, nearest hits out. This is Phase 1 of ADR-001 and it is the baseline the
// OptiX path will be measured against, so the two must agree hit for hit before any
// speedup number is allowed to mean something.
//
// Correctness is checked three ways, because a tracer that is subtly wrong produces
// plausible garbage forever:
//
//   1. analytic  -- rays at a plane of known range and tilt, where the answer is
//                   arithmetic and not an opinion
//   2. brute force -- every ray against every triangle, no hierarchy at all. If the
//                   BVH disagrees with this, the BVH is wrong. It is O(N*M) and only
//                   runs on small scenes, which is the point of small scenes.
//   3. watertightness -- rays fired exactly along shared triangle edges, where a naive
//                   intersector leaks and reports a miss through solid geometry
// =============================================================================

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

// -----------------------------------------------------------------------------
// Device-side types. Deliberately plain: the same structs the host builds.
// -----------------------------------------------------------------------------
struct Ray {
    float3 origin;
    float3 direction;   // expected normalised
    float  tmax;
};

struct Hit {
    float t;            // distance along the ray, or -1 for a miss
    float3 normal;      // geometric, unnormalised sign fixed against the ray
    int    triangle;
    int    material;
};

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float d_dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 d_cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

/// Moeller-Trumbore.
///
/// The epsilon is on the DETERMINANT and not on the barycentrics: a ray parallel to the
/// triangle's plane is the degenerate case, and a ray that merely grazes an edge is not.
/// Testing the barycentrics against an epsilon instead is the classic way to open a crack
/// along every shared edge in the mesh.
__device__ __forceinline__ bool intersect_triangle(
    const float3& o, const float3& d,
    const float3& v0, const float3& v1, const float3& v2,
    float tmax, float& t_out, float3& n_out)
{
    const float3 e1 = v1 - v0;
    const float3 e2 = v2 - v0;
    const float3 p  = d_cross(d, e2);
    const float det = d_dot(e1, p);

    if (fabsf(det) < 1e-12f) return false;      // ray parallel to the plane

    const float inv = 1.0f / det;
    const float3 tv = o - v0;

    // The barycentric bounds are relaxed by an epsilon, and the direction of that
    // relaxation is the whole point.
    //
    // A ray striking a vertex shared by several triangles lands exactly on every one of
    // their boundaries at once. Rounding can then put u or v a few ULP outside on ALL of
    // them, every triangle rejects, and the ray passes through solid geometry.
    //
    // Kept as a defence, not as a fix: this was the first explanation offered for the six
    // leaks the watertightness check found, and it was WRONG -- relaxing these bounds
    // changed nothing. The leaks were a NaN in the slab test, see safe_inverse below. The
    // epsilon stays because the failure mode it guards against is real and the cost is
    // nil, but it has not been observed to catch anything here.
    //
    // Widening instead of tightening means adjacent triangles may both accept a
    // boundary hit. That costs nothing: they report the same distance and the nearest-hit
    // logic keeps one. A leak is a hole in a surface that is supposed to be closed, and
    // for a LiDAR simulator holes appear precisely at edges and corners -- which is where
    // the multi-return physics this project exists to study actually lives.
    constexpr float kBary = 1e-6f;

    const float u = d_dot(tv, p) * inv;
    if (u < -kBary || u > 1.0f + kBary) return false;

    const float3 q = d_cross(tv, e1);
    const float v = d_dot(d, q) * inv;
    if (v < -kBary || u + v > 1.0f + kBary) return false;

    const float t = d_dot(e2, q) * inv;
    if (t <= 1e-6f || t >= tmax) return false;   // behind the origin, or further than a known hit

    t_out = t;
    n_out = d_cross(e1, e2);
    return true;
}

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
__device__ __forceinline__ float3 safe_inverse(const float3& d) {
    const float ex = fabsf(d.x) < 1e-20f ? copysignf(1e-20f, d.x) : d.x;
    const float ey = fabsf(d.y) < 1e-20f ? copysignf(1e-20f, d.y) : d.y;
    const float ez = fabsf(d.z) < 1e-20f ? copysignf(1e-20f, d.z) : d.z;
    return make_float3(1.0f / ex, 1.0f / ey, 1.0f / ez);
}

/// Slab test. Returns the near distance, or -1 when the ray misses the box.
///
/// Written with the reciprocal precomputed by the caller so the divide is not paid per
/// node -- traversal visits many nodes per ray and this is the inner loop.
__device__ __forceinline__ float intersect_aabb(
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

__global__ void trace_bvh(
    const BvhNode* __restrict__ nodes,
    const int* __restrict__ indices,
    const Triangle* __restrict__ tris,
    const Ray* __restrict__ rays,
    Hit* __restrict__ hits,
    int n_rays)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) return;

    const Ray r = rays[i];
    const float3 inv_d = safe_inverse(r.direction);

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
                if (intersect_triangle(r.origin, r.direction, v0, v1, v2, best_t, t, nrm)) {
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
    hits[i] = h;
}

/// Every ray against every triangle. The oracle: no hierarchy, nothing to get wrong
/// except the intersector itself, which the analytic test covers separately.
__global__ void trace_brute(
    const Triangle* __restrict__ tris, int n_tris,
    const Ray* __restrict__ rays, Hit* __restrict__ hits, int n_rays)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) return;

    const Ray r = rays[i];
    float best_t = r.tmax;
    int best_tri = -1;
    float3 best_n = make_float3(0.f, 0.f, 0.f);

    for (int ti = 0; ti < n_tris; ++ti) {
        const Triangle tr = tris[ti];
        const float3 v0 = make_float3(tr.v0.x, tr.v0.y, tr.v0.z);
        const float3 v1 = make_float3(tr.v1.x, tr.v1.y, tr.v1.z);
        const float3 v2 = make_float3(tr.v2.x, tr.v2.y, tr.v2.z);
        float t; float3 nrm;
        if (intersect_triangle(r.origin, r.direction, v0, v1, v2, best_t, t, nrm)) {
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
    hits[i] = h;
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
            const float x0 = -h + i * step, x1 = x0 + step;
            const float y0 = -h + j * step, y1 = y0 + step;
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
    int n_tris = 0;

    void upload(const Scene& s) {
        n_tris = static_cast<int>(s.triangles.size());
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
        cudaFree(nodes); cudaFree(indices); cudaFree(tris);
    }
};

static std::vector<Hit> run(const DeviceScene& ds, const std::vector<Ray>& rays, bool brute) {
    Ray* d_rays = nullptr; Hit* d_hits = nullptr;
    const int n = static_cast<int>(rays.size());
    CUDA_CHECK(cudaMalloc(&d_rays, n * sizeof(Ray)));
    CUDA_CHECK(cudaMalloc(&d_hits, n * sizeof(Hit)));
    CUDA_CHECK(cudaMemcpy(d_rays, rays.data(), n * sizeof(Ray), cudaMemcpyHostToDevice));

    const int block = 128, grid = (n + block - 1) / block;
    if (brute) trace_brute<<<grid, block>>>(ds.tris, ds.n_tris, d_rays, d_hits, n);
    else       trace_bvh<<<grid, block>>>(ds.nodes, ds.indices, ds.tris, d_rays, d_hits, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Hit> hits(n);
    CUDA_CHECK(cudaMemcpy(hits.data(), d_hits, n * sizeof(Hit), cudaMemcpyDeviceToHost));
    cudaFree(d_rays); cudaFree(d_hits);
    return hits;
}

static int failures = 0;
static void check(bool ok, const char* what, const std::string& detail = "") {
    std::printf("  %-46s %s%s%s\n", what, ok ? "PASS" : "FAIL",
                detail.empty() ? "" : "  ", detail.c_str());
    if (!ok) failures++;
}

int main() {
    std::printf("ARGOS Phase 1 -- CUDA BVH traversal\n\n");

    // ---- 1. analytic: a tilted plane has a closed-form hit distance ----------
    {
        const float d = 10.0f, tilt = 30.0f;
        Scene s = make_tilted_plane(d, tilt);
        DeviceScene ds; ds.upload(s);

        std::vector<Ray> rays;
        std::vector<float> expected;
        const float ta = std::tan(tilt * 3.14159265358979f / 180.f);
        for (int i = -8; i <= 8; ++i) {
            const float x = i * 0.5f;
            rays.push_back({make_float3(x, 0.f, 0.f), make_float3(0.f, 0.f, 1.f), 1e30f});
            expected.push_back(d + x * ta);   // plane: z = d + x*tan(a)
        }

        const auto hits = run(ds, rays, false);
        float worst = 0.f;
        for (size_t i = 0; i < rays.size(); ++i)
            worst = std::max(worst, std::fabs(hits[i].t - expected[i]));

        check(worst < 1e-3f, "analytic tilted plane, 17 rays",
              "max |error| = " + std::to_string(worst) + " m");

        // cos(theta) between the ray and the surface normal must equal cos(tilt).
        const float expect_cos = std::cos(tilt * 3.14159265358979f / 180.f);
        float worst_cos = 0.f;
        for (const auto& h : hits) {
            const float c = std::fabs(h.normal.z);        // ray is along +z
            worst_cos = std::max(worst_cos, std::fabs(c - expect_cos));
        }
        check(worst_cos < 1e-4f, "surface normal matches the known tilt",
              "max |error| = " + std::to_string(worst_cos));
        ds.free();
    }

    // ---- 2. the BVH must agree with brute force, hit for hit ----------------
    {
        Scene s = make_box(4.0f, 6);
        DeviceScene ds; ds.upload(s);
        std::printf("  [box: %zu triangles, %zu BVH nodes]\n", s.triangles.size(), s.nodes.size());

        // STRUCTURAL CHECK, and it exists because its absence let a real bug through.
        //
        // An empty AABB poisoned the SAH accumulator, so every split costed as infinite
        // and the builder returned ONE node holding all 432 triangles. The agreement test
        // below still passed -- a single-leaf "hierarchy" IS brute force, so the two
        // tracers agreed perfectly while measuring nothing. A test a broken
        // implementation passes is not a test.
        size_t leaves = 0, biggest_leaf = 0, in_leaves = 0;
        for (const auto& n : s.nodes)
            if (n.count > 0) {
                leaves++;
                in_leaves += n.count;
                biggest_leaf = std::max(biggest_leaf, static_cast<size_t>(n.count));
            }
        check(s.nodes.size() > 1, "the BVH actually branches",
              std::to_string(s.nodes.size()) + " nodes, " + std::to_string(leaves) + " leaves");
        check(biggest_leaf <= 4, "no leaf exceeds the maximum",
              "largest holds " + std::to_string(biggest_leaf));
        check(in_leaves == s.triangles.size(), "every triangle referenced exactly once",
              std::to_string(in_leaves) + " of " + std::to_string(s.triangles.size()));

        // Rays from outside in every direction, deterministic so a failure reproduces.
        std::vector<Ray> rays;
        unsigned seed = 12345u;
        auto rnd = [&]() {
            seed = seed * 1664525u + 1013904223u;
            return (seed >> 8) * (1.0f / 16777216.0f);     // [0,1)
        };
        for (int i = 0; i < 20000; ++i) {
            const float th = rnd() * 6.283185f, ph = std::acos(1.f - 2.f * rnd());
            const Vec3 o{8.f * std::sin(ph) * std::cos(th), 8.f * std::sin(ph) * std::sin(th),
                         8.f * std::cos(ph)};
            const Vec3 target{(rnd() - 0.5f) * 3.f, (rnd() - 0.5f) * 3.f, (rnd() - 0.5f) * 3.f};
            const Vec3 dir = normalize(target - o);
            rays.push_back({make_float3(o.x, o.y, o.z), make_float3(dir.x, dir.y, dir.z), 1e30f});
        }

        const auto a = run(ds, rays, false);
        const auto b = run(ds, rays, true);

        int mismatched = 0; float worst_t = 0.f;
        for (size_t i = 0; i < rays.size(); ++i) {
            const bool hit_a = a[i].t >= 0.f, hit_b = b[i].t >= 0.f;
            if (hit_a != hit_b) { mismatched++; continue; }
            if (hit_a) worst_t = std::max(worst_t, std::fabs(a[i].t - b[i].t));
        }
        check(mismatched == 0, "BVH agrees with brute force on 20000 rays",
              std::to_string(mismatched) + " disagreements");
        check(worst_t < 1e-4f, "distances identical where both hit",
              "max |delta| = " + std::to_string(worst_t) + " m");
        ds.free();
    }

    // ---- 3. watertightness: no ray may escape a closed box from inside -------
    {
        Scene s = make_box(4.0f, 8);
        DeviceScene ds; ds.upload(s);

        std::vector<Ray> rays;
        unsigned seed = 999u;
        auto rnd = [&]() {
            seed = seed * 1664525u + 1013904223u;
            return (seed >> 8) * (1.0f / 16777216.0f);
        };
        // Fired from the exact centre, including straight down the axes and diagonals,
        // which are where a mesh's shared edges line up with the ray.
        for (int i = 0; i < 30000; ++i) {
            float dx, dy, dz;
            if (i < 6) {                                   // the six axis directions
                dx = (i == 0) - (i == 1); dy = (i == 2) - (i == 3); dz = (i == 4) - (i == 5);
            } else if (i < 14) {                           // the eight corner diagonals
                const int b = i - 6;
                const float k = 0.57735027f;
                dx = (b & 1) ? k : -k; dy = (b & 2) ? k : -k; dz = (b & 4) ? k : -k;
            } else {
                const float th = rnd() * 6.283185f, ph = std::acos(1.f - 2.f * rnd());
                dx = std::sin(ph) * std::cos(th); dy = std::sin(ph) * std::sin(th); dz = std::cos(ph);
            }
            rays.push_back({make_float3(0.f, 0.f, 0.f), make_float3(dx, dy, dz), 1e30f});
        }

        const auto hits = run(ds, rays, false);
        int escaped = 0;
        for (const auto& h : hits) if (h.t < 0.f) escaped++;
        check(escaped == 0, "no ray escapes a closed box (30000 rays)",
              std::to_string(escaped) + " leaks");
        ds.free();
    }

    std::printf("\n%s\n", failures == 0 ? "all checks passed" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
