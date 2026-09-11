// =============================================================================
// QuBLAR -- multi-bounce transport, and the three-bounce path that sees around corners
// =============================================================================
// ADR-004 item 2. Everything before this file assumed light leaves the sensor, hits one
// surface and comes back. Non-line-of-sight imaging breaks that assumption at its root:
// the photons carrying information about a hidden object never travel in a straight line
// to it.
//
//     sensor -> p (relay wall) -> q (hidden) -> p -> sensor
//
// Confocal: the beam illuminates only p, and the detector collects only from p. Under
// that assumption the hidden object is never directly lit and a direct return from it
// could not be collected anyway, which is why no occluder is modelled -- the confocal
// optics already enforce what an occluder enforces in a real rig. That is a consequence,
// not an omission, and it is written down because "we forgot the occluder" and "the
// occluder is redundant here" look identical in the output.
//
// The direct return off the relay wall is kept rather than suppressed. It is present in
// real data, it has to be gated out by whoever processes it, and it gives validation a
// second arrival whose time is exactly 2|p - s|/c at an amplitude orders of magnitude
// above the hidden-object signal.
// =============================================================================

#pragma once

#include "lidar.cuh"

namespace argos {

struct TransientConfig {
    /// Paths traced per relay point. This is a Monte Carlo integral over the hemisphere
    /// at the relay wall, and the hidden object subtends a small solid angle, so this
    /// number is what decides whether the three-bounce return is visible at all.
    int paths = 65536;

    /// Albedo of the relay wall. Kept separate from the scene's materials because the
    /// wall is not "a surface in the scene" to this model -- it is the aperture through
    /// which the hidden volume is observed.
    float wall_albedo = 0.8f;

    /// Relative strength of the first-bounce wall return. Real systems see it saturate the
    /// detector; here it is simply recorded at its natural amplitude of 1.
    bool include_direct = true;

    unsigned seed = 0x51ED270Bu;
};

/// One relay point on the wall, with the local frame the hemisphere is sampled in.
struct RelayPoint {
    float3 position;
    float3 normal;      // unit, pointing into the hidden half-space
};

/// Cosine-weighted direction over the hemisphere about `n`.
///
/// Cosine-weighted rather than uniform because the Lambertian factor then cancels exactly
/// against the pdf, leaving the wall albedo alone. Uniform sampling would be correct too
/// and would put a cosine in the numerator and a constant in the denominator, converging
/// to the same answer more slowly and with more places to put the factor in the wrong
/// place.
__device__ __forceinline__ float3 cosine_hemisphere(const float3& n, float u1, float u2) {
    const float r = sqrtf(u1);
    const float phi = 6.28318530718f * u2;
    const float x = r * cosf(phi);
    const float y = r * sinf(phi);
    const float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    float3 t, b;
    onb(n, t, b);
    return t * x + b * y + n * z;
}

/// Trace the transient response of one relay point per block.
///
/// The per-path weight is derived rather than copied:
///
///   - the first bounce contributes f_p*cos(theta_p)/pdf = (rho_w/pi * cos)/(cos/pi)
///     = rho_w, so the sampling cosine cancels exactly;
///   - q is Lambertian and returns rho_q/pi * cos(theta_q) toward p;
///   - the receiver at p subtends A_p*cos(theta_p)/r^2.
///
///       weight  =  rho_w * rho_q * cos(theta_q) * cos(theta_p) / r^2
///
/// This does NOT contradict the 1/r^4 the NLOS literature quotes. That figure is per unit
/// AREA of hidden surface, and converting solid angle to area brings a second
/// cos(theta_q)/r^2. Same model, different thing held fixed: sampling solid angle carries
/// 1/r^2 per path and produces 1/r^4 per unit area on its own, which is what the falloff
/// check measures.
__global__ void transient_trace(
    const BvhNode* __restrict__ nodes,
    const int* __restrict__ indices,
    const Triangle* __restrict__ tris,
    const Material* __restrict__ mats,
    const RelayPoint* __restrict__ relays,
    float3 sensor,
    SensorConfig scfg,
    TransientConfig tcfg,
    float* __restrict__ transients,     // n_relays * bins
    int n_relays)
{
    const int rp = blockIdx.x;
    if (rp >= n_relays) return;

    extern __shared__ float wave[];
    for (int k = threadIdx.x; k < scfg.bins; k += blockDim.x) wave[k] = 0.0f;
    __syncthreads();

    const RelayPoint relay = relays[rp];
    const float sigma = pulse_sigma(scfg);
    const float c = static_cast<float>(kSpeedOfLight);

    const float3 to_wall = relay.position - sensor;
    const float L1 = sqrtf(d_dot(to_wall, to_wall));

    // --- the first-bounce return off the wall itself --------------------------
    if (tcfg.include_direct && threadIdx.x == 0)
        splat_pulse(wave, scfg, 2.0f * L1 / c, 1.0f, sigma);

    const float inv_paths = 1.0f / tcfg.paths;

    for (int i = threadIdx.x; i < tcfg.paths; i += blockDim.x) {
        unsigned key = hash_u32(static_cast<unsigned>(rp) * 0x9E3779B9u
                              ^ static_cast<unsigned>(i) * 0x85EBCA6Bu ^ tcfg.seed);
        const float u1 = (hash_u32(key) >> 8) * (1.0f / 16777216.0f);
        const float u2 = (hash_u32(key ^ 0x68BC21EBu) >> 8) * (1.0f / 16777216.0f);

        const float3 dir = cosine_hemisphere(relay.normal, u1, u2);

        Ray r;
        // Offset along the normal, not along the ray. Offsetting along the ray leaves a
        // grazing path still inside the surface it started on, which is how a relay wall
        // starts shadowing itself and the transient grows a return at t = 2*L1/c that is
        // not the wall return.
        r.origin = relay.position + relay.normal * 1e-4f;
        r.direction = dir;
        r.tmax = scfg.max_range();

        const Hit h = traverse_bvh(nodes, indices, tris, r);
        if (h.t <= 0.0f) continue;

        const float rr = h.t;
        const float cos_q = fabsf(d_dot(h.normal, dir));
        const float cos_p = fabsf(d_dot(relay.normal, dir));
        const float rho_q = mats[h.material].reflectance;

        const float weight = tcfg.wall_albedo * rho_q * cos_q * cos_p / (rr * rr);

        // Total path length: out to the wall, out to q, back to the wall, back to the
        // sensor. Confocal, so the two wall legs are the same length and the two hidden
        // legs are the same length.
        const float t_arrive = 2.0f * (L1 + rr) / c;
        splat_pulse(wave, scfg, t_arrive, weight * inv_paths, sigma);
    }
    __syncthreads();

    for (int k = threadIdx.x; k < scfg.bins; k += blockDim.x)
        transients[static_cast<size_t>(rp) * scfg.bins + k] = wave[k];
}

inline size_t transient_smem_bytes(const SensorConfig& cfg) {
    return sizeof(float) * static_cast<size_t>(cfg.bins);
}

}  // namespace argos
