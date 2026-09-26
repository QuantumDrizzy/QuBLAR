// =============================================================================
// QuBLAR -- muon mode: shadow imaging a voxelized pyramid with cosmic rays
// =============================================================================
// ADR-006 item 1. The probe changes, the pipeline shape does not: transport,
// counting, truth alongside the measurement, and reconstructions scored
// against it.
//
//     sky (cos^2 theta) -> muon -> voxelized medium -> Beer-Lambert survival
//     -> direction-binned counts -> transmission MLEM -> density map
//
// The physics is deliberately the counting model every transmission muography
// analysis reduces to: a muon along ray r survives with probability
// exp(-tau_r), tau_r = int mu(s) ds, with the energy spectrum folded into one
// declared effective coefficient (ADR-006 §1). This simulates the statistics
// of shadow imaging, not the cascade physics inside it.
//
// The marcher is written __host__ __device__ ONCE and is used by the GPU
// exposure and by the host reconstruction -- two implementations of one
// marcher is the duplication that cost Phase 2 a confound (ADR-006 §3).
//
// The axis-aligned ray is handled explicitly: an infinite inverse direction
// component combined with a zero boundary distance is the 0*inf = NaN trap
// that already sailed rays through solid geometry once in this repository.
// =============================================================================

#pragma once

#include "lidar.cuh"

#include <cmath>

namespace argos {

// -----------------------------------------------------------------------------
// The medium: a uniform voxel grid of attenuation coefficients
// -----------------------------------------------------------------------------

struct VoxelMedium {
    float3 lo;              // grid min corner, metres
    float  voxel;           // isotropic voxel edge, metres
    int    nx, ny, nz;
    float  mu_rock, mu_air; // effective attenuation, 1/m
    const float* mu;        // per-voxel coefficient [nx*ny*nz]; host pointer
                            // when marching on the host, device pointer when
                            // marching on the device

    __host__ __device__ __forceinline__ int index(int i, int j, int k) const {
        return (k * ny + j) * nx + i;
    }
    __host__ __device__ __forceinline__ float mu_at(int i, int j, int k) const {
        return mu[index(i, j, k)];
    }
    __host__ __device__ bool inside(float3 p) const {
        return p.x >= lo.x && p.x < lo.x + nx * voxel
            && p.y >= lo.y && p.y < lo.y + ny * voxel
            && p.z >= lo.z && p.z < lo.z + nz * voxel;
    }
};

/// The one Amanatides-Woo stepping, shared by transport and reconstruction
/// (ADR-006 §3). `visit(index, segment_length)` is called per voxel crossed
/// and returns that segment's contribution to the returned total -- the
/// transport accumulates mu*seg, the reconstruction records segments. A
/// functor rather than a device lambda so the same template instantiates on
/// both sides without --extended-lambda.
template <typename Visit>
__host__ __device__ __forceinline__ float march_impl(const VoxelMedium& m,
                                                     float3 p, float3 d,
                                                     Visit visit) {
    // grid AABB entry/exit, with the axis-aligned guard: an infinite inverse
    // component must never be multiplied by a zero distance.
    const float eps = 1e-12f;
    float t0 = -1e30f, t1 = 1e30f;
    const float pc[3]  = {p.x, p.y, p.z};
    const float dc[3]  = {d.x, d.y, d.z};
    const float loc[3] = {m.lo.x, m.lo.y, m.lo.z};
    const int   nc[3]  = {m.nx, m.ny, m.nz};
    for (int a = 0; a < 3; ++a) {
        if (fabsf(dc[a]) < eps) {
            if (pc[a] < loc[a] || pc[a] > loc[a] + nc[a] * m.voxel) return 0.0f;
        } else {
            const float ta = (loc[a] - pc[a]) / dc[a];
            const float tb = (loc[a] + nc[a] * m.voxel - pc[a]) / dc[a];
            t0 = fmaxf(t0, fminf(ta, tb));
            t1 = fminf(t1, fmaxf(ta, tb));
        }
    }
    if (t1 <= fmaxf(t0, 0.0f)) return 0.0f;
    const float t_start = fmaxf(t0, 0.0f);
    const float t_end = t1 - t_start;
    const float3 q = p + d * t_start;

    // Amanatides-Woo state. Voxel indices clamped: q may sit exactly on the
    // entry boundary (floor gives -1 or n there).
    int vox[3];
    float tMax[3], tDelta[3];
    vox[0] = static_cast<int>(floorf((q.x - m.lo.x) / m.voxel));
    vox[1] = static_cast<int>(floorf((q.y - m.lo.y) / m.voxel));
    vox[2] = static_cast<int>(floorf((q.z - m.lo.z) / m.voxel));
    for (int a = 0; a < 3; ++a) {
        if (vox[a] < 0) vox[a] = 0;
        if (vox[a] >= nc[a]) vox[a] = nc[a] - 1;
        if (fabsf(dc[a]) < eps) {
            tMax[a] = 1e30f;
            tDelta[a] = 0.0f;
        } else {
            const float boundary = (dc[a] > 0.0f ? float(vox[a] + 1) : float(vox[a]))
                                   * m.voxel + loc[a];
            tMax[a] = (boundary - pc[a]) / dc[a];
            tDelta[a] = m.voxel / fabsf(dc[a]);
            if (tMax[a] < 0.0f) tMax[a] = 0.0f;
        }
    }

    float tau = 0.0f, t_done = 0.0f;
    // Bounded: a monotone march visits at most nx+ny+nz voxels; the bound is
    // slack for safety, so a stepping bug shows as a wrong answer rather than
    // as a hang.
    for (int steps = 0; steps < 3 * (m.nx + m.ny + m.nz) + 16; ++steps) {
        const int axis = (tMax[0] <= tMax[1])
            ? ((tMax[0] <= tMax[2]) ? 0 : 2)
            : ((tMax[1] <= tMax[2]) ? 1 : 2);
        const float t_next = fminf(tMax[axis], t_end);
        const float seg = t_next - t_done;
        if (seg > 0.0f)
            tau += visit(m.index(vox[0], vox[1], vox[2]), seg);
        if (t_next >= t_end) break;
        t_done = t_next;
        vox[axis] += (dc[axis] > 0.0f) ? 1 : -1;
        tMax[axis] += tDelta[axis];
        if (vox[axis] < 0 || vox[axis] >= nc[axis]) break;   // marched out
    }
    return tau;
}

/// Transport wrapper: optical depth = sum of mu * segment.
struct TauAccum {
    const VoxelMedium* m;
    __host__ __device__ __forceinline__ float operator()(int idx, float seg) const {
        return m->mu[idx] * seg;
    }
};

__host__ __device__ __forceinline__ float march_medium(const VoxelMedium& m,
                                                       float3 p, float3 d) {
    return march_impl(m, p, d, TauAccum{&m});
}

// -----------------------------------------------------------------------------
// The sky: dN/dOmega proportional to cos^2 theta, zenith-capped
// -----------------------------------------------------------------------------

/// Sample one sky direction. Closed form: with c = cos theta, p(c) ∝ c^2 on
/// [c_min, 1], so c = (c_min^3 + u(1 - c_min^3))^(1/3) -- no rejection loop.
/// `vertical` is the closed-form-check mode (ADR-006 §5): straight up.
__host__ __device__ __forceinline__ void sample_sky(bool vertical, float c_min,
                                                    unsigned key,
                                                    float3& d, float& cos_th,
                                                    float& az) {
    if (vertical) {
        d = make_float3(0.0f, 0.0f, 1.0f);
        cos_th = 1.0f;
        az = 0.0f;
        return;
    }
    const float u1 = hash_unit(key);
    const float u2 = hash_unit(key ^ 0x68BC21EBu);
    const float c3 = c_min * c_min * c_min;
    const float c = cbrtf(c3 + u1 * (1.0f - c3));
    const float s = sqrtf(fmaxf(0.0f, 1.0f - c * c));
    az = 6.28318530718f * u2;
    d = make_float3(s * cosf(az), s * sinf(az), c);
    cos_th = c;
}

// -----------------------------------------------------------------------------
// Exposure kernels
// -----------------------------------------------------------------------------

struct MuonSky {
    bool  vertical = false;
    float c_min = 0.34202f;   // cos(70 deg): the zenith cap
    float th_max = 1.22173f;  // 70 deg, radians
    int   n_az = 48;          // azimuth bins over [0, 2pi)
    int   n_th = 20;          // zenith bins over [0, th_max]

    __host__ __device__ int size() const { return n_th * n_az; }
};

/// One candidate muon per thread: sample the sky, bin it, march the medium
/// from the chamber, detect with Beer-Lambert. Open counts are the candidate
/// population per bin; detected counts are the shadow. Hash-sampled, so the
/// candidate population is deterministic per index and resampling detection
/// alone gives exact Binomial(N, T) batch statistics (ADR-006 §5).
__global__ void muon_expose_binned(const VoxelMedium m, MuonSky sky,
                                   int n_candidates, float3 chamber,
                                   unsigned seed,
                                   unsigned long long* open_counts,
                                   unsigned long long* det_counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_candidates) return;

    // Directions are keyed by the candidate index ALONE: the candidate
    // population is identical across exposures, and resampling the detection
    // (the seed below) then gives exact Binomial(N, T) batch statistics.
    const unsigned key = hash_u32(static_cast<unsigned>(i) * 0x9E3779B9u);
    float3 d; float c, az;
    sample_sky(sky.vertical, sky.c_min, key, d, c, az);

    const float th = acosf(fminf(1.0f, fmaxf(-1.0f, c)));
    int ith = static_cast<int>(th / sky.th_max * sky.n_th);
    if (ith < 0) ith = 0;
    if (ith >= sky.n_th) ith = sky.n_th - 1;
    int iaz = static_cast<int>(az * 0.15915494f * sky.n_az);  // az / (2 pi)
    if (iaz < 0) iaz = 0;
    if (iaz >= sky.n_az) iaz = sky.n_az - 1;
    const int bin = ith * sky.n_az + iaz;

    atomicAdd(&open_counts[bin], 1ULL);

    const float tau = march_medium(m, chamber, d);
    const float u3 = hash_unit(key ^ 0x2545F491u ^ seed);
    if (u3 < expf(-tau)) atomicAdd(&det_counts[bin], 1ULL);
}

/// Vertical rays through a horizontal line of pixel positions along +x: the
/// closed-form-check geometry. One block per pixel, per-thread Bernoulli
/// detection, and the marched tau recorded once as per-ray truth.
__global__ void muon_vertical(const VoxelMedium m, int n_samples, int n_pix,
                              float b_max, float det_z, unsigned seed,
                              unsigned* det_counts, float* tau_true) {
    const int pix = blockIdx.x;
    if (pix >= n_pix) return;
    const float b = b_max * (pix + 0.5f) / n_pix;
    const float3 origin = make_float3(b, 0.0f, det_z);
    const float3 dir = make_float3(0.0f, 0.0f, 1.0f);

    // Marched once per pixel, shared with the block: a plain global write by
    // thread 0 read by the rest would be a race, and a race between "tau" and
    // "no tau" looks exactly like extra transmission.
    __shared__ float s_tau;
    if (threadIdx.x == 0) s_tau = march_medium(m, origin, dir);
    __syncthreads();
    const float p_det = expf(-s_tau);
    int hits = 0;
    for (int s = threadIdx.x; s < n_samples; s += blockDim.x) {
        const unsigned key = hash_u32(static_cast<unsigned>(pix) * 0x85EBCA6Bu
                                    ^ static_cast<unsigned>(s) * 0xC2B2AE35u
                                    ^ seed);
        if (hash_unit(key) < p_det) ++hits;
    }
    // Block reduction of per-thread hits, then one atomic per block. The
    // caller launches exactly 256 threads per block (the reduction is written
    // for that width).
    __shared__ unsigned red[256];
    red[threadIdx.x] = static_cast<unsigned>(hits);
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&det_counts[pix], red[0]);
        tau_true[pix] = s_tau;
    }
}

/// Bytes-independent helper: grid launch sizing used by the callers.
inline int muon_grid(int n_threads, int block) {
    return (n_threads + block - 1) / block;
}

}  // namespace argos
