// =============================================================================
// QuBLAR -- the types both tracers speak
// =============================================================================
// ADR-002 requires the CUDA baseline and the OptiX path to produce identical output
// for identical input. That is only checkable if they share one definition of what a
// ray and a hit are, rather than two that happen to look alike today.
// =============================================================================

#pragma once

#include <cuda_runtime.h>
#include <cmath>

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

__host__ __device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ __forceinline__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ __forceinline__ float d_dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ __forceinline__ float3 d_cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

