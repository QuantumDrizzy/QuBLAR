// =============================================================================
// QuBLAR -- the RT-core path, as OptiX device programs
// =============================================================================
// ADR-002 item 2. Same rays in, same Hit out as traverse_bvh in trace.cuh. The only
// difference is who walks the hierarchy: here it is fixed-function silicon, there it
// is a stack in registers.
//
// There is no payload beyond what OptiX needs to reach the hit programs. The closest-hit
// and miss programs write the output buffer directly at the launch index, which keeps
// payload registers at zero and makes the data path identical to the CUDA kernel's:
// one store per ray, no intermediate.
//
// Compiled to OptiX-IR, never PTX -- CUDA 13 validates --ptx output with ptxas, and
// ptxas does not know the OptiX intrinsics.
// =============================================================================

#include <optix.h>
#include "optix_shared.h"

extern "C" {
__constant__ OptixParams params;
}

// -----------------------------------------------------------------------------

static __forceinline__ __device__ float3 d_sub(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static __forceinline__ __device__ float d_dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static __forceinline__ __device__ float3 d_cross3(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

// -----------------------------------------------------------------------------

extern "C" __global__ void __raygen__trace() {
    const unsigned i = optixGetLaunchIndex().x;
    if (i >= static_cast<unsigned>(params.n_rays)) return;

    const Ray r = params.rays[i];

    // tmin matches the baseline's near clip exactly. The CUDA intersector rejects
    // t <= 1e-6, so an OptiX path that accepted t = 0 would disagree with it on
    // self-intersection and the disagreement would be the harness's fault, not the
    // hardware's.
    optixTrace(params.handle,
               r.origin, r.direction,
               1e-6f, r.tmax,
               0.0f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               0, 1, 0);
}

extern "C" __global__ void __miss__trace() {
    const unsigned i = optixGetLaunchIndex().x;
    Hit h;
    h.t = -1.0f;
    h.normal = make_float3(0.f, 0.f, 0.f);
    h.triangle = -1;
    h.material = -1;
    params.hits[i] = h;
}

extern "C" __global__ void __closesthit__trace() {
    const unsigned i = optixGetLaunchIndex().x;
    const int prim = optixGetPrimitiveIndex();

    // The normal is recomputed from the vertex buffer rather than taken from the
    // barycentrics, using cross(v1 - v0, v2 - v0) -- the same expression and the same
    // operand order as intersect_triangle_wt. Anything else would make a normal mismatch
    // between the two paths ambiguous: hardware disagreement, or two different formulas?
    const float3 v0 = params.verts[3 * prim + 0];
    const float3 v1 = params.verts[3 * prim + 1];
    const float3 v2 = params.verts[3 * prim + 2];
    float3 n = d_cross3(d_sub(v1, v0), d_sub(v2, v0));

    const float3 dir = optixGetWorldRayDirection();
    const float len = sqrtf(d_dot3(n, n));
    if (len > 0.f) n = make_float3(n.x / len, n.y / len, n.z / len);
    if (d_dot3(n, dir) > 0.f) n = make_float3(-n.x, -n.y, -n.z);

    Hit h;
    h.t = optixGetRayTmax();
    h.normal = n;
    h.triangle = prim;
    h.material = params.materials[prim];
    params.hits[i] = h;
}
