// =============================================================================
// QuBLAR -- the sensor model: beam, radiometry, waveform, and ground truth
// =============================================================================
// ADR-001 items 3-6. A beam is not a ray: it has angular divergence, so its footprint
// at range covers an area that may contain several surfaces at different distances.
// That is the entire reason this project exists -- a single-return range image cannot
// express what a full-waveform instrument records, and the interesting physics lives
// where the footprint straddles an edge.
//
// One CUDA block per beam. The waveform lives in shared memory for the duration, so
// accumulation is a shared atomicAdd rather than a global one; a beam's samples all
// land in the same few hundred bins, and doing that in DRAM would make the memory
// system the experiment.
//
// Every measurement is emitted alongside the TRUTH that produced it: the distinct
// surfaces the footprint actually struck, with their ranges, incidence angles and
// weight fractions. Reconstruction is then scored against that rather than against
// another algorithm's output.
// =============================================================================

#pragma once

#include "trace.cuh"

namespace argos {

/// Exact, not 3e8: the range scale of every number in this file is set by it, and at
/// 100 m the rounded value is off by 7 cm -- larger than a range bin.
constexpr double kSpeedOfLight = 299792458.0;

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

/// One emitted beam. The axis is the nominal pointing direction; real energy is spread
/// around it by divergence_half_angle.
struct Beam {
    float3 origin;
    float3 axis;                    // normalised
    float  divergence_half_angle;   // radians, the 1/e^2 half-angle of the Gaussian beam
};

/// A time-resolved return, one surface's worth of truth.
struct TruthReturn {
    float range;        // metres, weight-averaged over the samples in this cluster
    float cos_theta;    // weight-averaged incidence
    float weight;       // fraction of the beam's energy that struck this surface, [0, 1]
    int   material;
    int   samples;      // footprint samples that landed here
};

struct SensorConfig {
    // --- time gate ---
    int   bins          = 1024;
    float bin_seconds   = 0.2e-9f;   // 0.2 ns -> 3.0 cm of range per bin
    float t0_seconds    = 0.0f;

    // --- transmitted pulse ---
    float pulse_fwhm_seconds = 1.0e-9f;

    // --- footprint sampling ---
    int   samples_sqrt  = 16;        // the footprint is sampled on an S x S stratified grid
    unsigned seed       = 0x9E3779B9u;

    // --- ground truth ---
    int   max_returns   = 4;
    float truth_merge_m = 0.05f;     // surfaces closer than this are one surface

    __host__ __device__ int   samples() const { return samples_sqrt * samples_sqrt; }
    __host__ __device__ float gate_end() const { return t0_seconds + bins * bin_seconds; }
    /// Longest range the gate can represent. Rays are cut here: a return arriving after
    /// the gate closes is not recorded by a real instrument either.
    __host__ __device__ float max_range() const {
        return 0.5f * gate_end() * static_cast<float>(kSpeedOfLight);
    }
};

/// Gaussian sigma of the transmitted pulse, from its full width at half maximum.
__host__ __device__ __forceinline__ float pulse_sigma(const SensorConfig& c) {
    return c.pulse_fwhm_seconds / 2.354820045f;   // 2*sqrt(2*ln 2)
}

// -----------------------------------------------------------------------------
// Sampling
// -----------------------------------------------------------------------------

/// Wang hash. Deterministic per (beam, sample), which is the requirement: the CPU
/// mirror in the validation suite has to reproduce the GPU sample set exactly, and a
/// stateful RNG whose sequence depends on thread scheduling cannot promise that.
__host__ __device__ __forceinline__ unsigned hash_u32(unsigned x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15);
    return x;
}

__host__ __device__ __forceinline__ float hash_unit(unsigned x) {
    return (hash_u32(x) >> 8) * (1.0f / 16777216.0f);   // [0, 1)
}

/// Build an orthonormal basis around n without a branch on the dominant axis
/// (Duff et al. 2017). The naive version picks a helper axis with an if, and produces a
/// basis that flips discontinuously -- which shows up as a seam in the sampled
/// footprint precisely at the pointing directions a scanner uses most.
__host__ __device__ __forceinline__ void onb(const float3& n, float3& t, float3& b) {
    const float sign = n.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (sign + n.z);
    const float c = n.x * n.y * a;
    t = make_float3(1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x);
    b = make_float3(c, sign + n.y * n.y * a, -n.y);
}

/// One footprint sample: a direction within the cone and the share of the beam energy
/// it carries.
///
/// Directions are drawn uniformly in SOLID ANGLE, not uniformly in angle. Uniform in
/// angle oversamples the axis, which biases the estimator toward whatever the centre of
/// the beam happens to hit -- exactly the error that would make an edge-straddling
/// footprint report a clean single return.
///
/// The weight is the Gaussian beam profile evaluated at the off-axis angle, with the
/// divergence half-angle taken as the 1/e^2 point. Weights are normalised later by their
/// sum over ALL samples, including those that hit nothing: a sample that misses still
/// carried beam energy, it simply brought none back. Normalising by the hitting samples
/// alone would hand a sub-footprint target the full return of an extended one and destroy
/// the 1/d^4 behaviour that small targets are supposed to show.
__host__ __device__ __forceinline__ void sample_beam(
    const Beam& beam, const SensorConfig& cfg, int beam_id, int s,
    float3& dir, float& weight)
{
    const int S = cfg.samples_sqrt;
    const int sx = s % S, sy = s / S;

    // Stratified: one sample per cell of an S x S grid, jittered inside its cell. Plain
    // independent sampling has the same mean and several times the variance, and variance
    // here is indistinguishable from a second surface in the waveform.
    const unsigned key = (static_cast<unsigned>(beam_id) * 0x85EBCA6Bu)
                       ^ (static_cast<unsigned>(s) * 0xC2B2AE35u) ^ cfg.seed;
    const float j1 = hash_unit(key);
    const float j2 = hash_unit(key ^ 0x68BC21EBu);
    const float xi1 = (sx + j1) / S;
    const float xi2 = (sy + j2) / S;

    // Uniform in solid angle means cos(alpha) = 1 - xi * (1 - cos(alpha_max)), and
    // writing it that way in float32 destroys the sampler at the divergences a LiDAR
    // actually uses.
    //
    // The spacing of float32 near 1.0 is 6e-8. For a 2 mrad half-angle, 1 - cos(a_max) is
    // 2e-6, so the subtraction keeps about five bits: the quantity carries ~3% relative
    // error and cos_a lands on roughly 34 distinct values. The footprint collapses into
    // concentric rings, and the innermost ring rounds to cos_a = 1 exactly -- some 1.5% of
    // the samples, carrying the LARGEST weights, all pointing precisely along the axis.
    //
    // Every energy invariant passes through that untouched, because it is symmetric in
    // range and in incidence. It was the half-plane symmetry check that caught it: those
    // axis-exact samples land on the boundary of the illuminated half and are accepted by
    // the intersector's edge epsilon, so a footprint split exactly in two returned 51.7%
    // of its energy instead of 50%, and did not converge with sample count because the
    // error was bias rather than variance.
    //
    // The versine form, 1 - cos(a) = 2 sin^2(a/2), has no cancellation: sin(1e-3) is
    // exact to full float precision, and alpha = 2 asin(sqrt(versin/2)) recovers the angle
    // without ever forming a number near 1.
    const float sh          = sinf(0.5f * beam.divergence_half_angle);
    const float versin_max  = 2.0f * sh * sh;
    const float versin      = xi1 * versin_max;
    const float alpha       = 2.0f * asinf(fminf(1.0f, sqrtf(0.5f * versin)));
    const float sin_a       = sinf(alpha);
    const float cos_a       = cosf(alpha);
    const float phi         = 6.28318530718f * xi2;

    float3 t, b;
    onb(beam.axis, t, b);
    dir = t * (sin_a * cosf(phi)) + b * (sin_a * sinf(phi)) + beam.axis * cos_a;

    const float amax = fmaxf(beam.divergence_half_angle, 1e-12f);
    const float r    = alpha / amax;
    weight = expf(-2.0f * r * r);
}

// -----------------------------------------------------------------------------
// Radiometry
// -----------------------------------------------------------------------------

/// Power returned by one footprint sample, before the pulse shape is applied.
///
/// The derivation, because a hand-rolled range equation is wrong about as often as it is
/// right. Irradiance at the target falls as 1/d^2. A Lambertian surface re-emits with
/// radiance rho*I*cos(theta)/pi. The receiver is colocated with the emitter, so it sees
/// the patch foreshortened by another cos(theta), while the patch subtended by a fixed
/// solid angle has area d^2*dw/cos(theta). Two of the three cosines cancel against the
/// area term, and the d^2 in that area cancels one of the two in the irradiance:
///
///     contribution  =  w * rho * cos(theta) / d^2
///
/// One cosine survives, which is the standard extended-target result and is what the
/// cos(theta) invariant checks. The 1/d^4 of a SUB-FOOTPRINT target is not written here
/// and must not be: it emerges on its own, because as range grows a small target
/// intercepts a falling fraction of the samples. A simulator that hard-codes 1/d^4 gets
/// extended targets wrong, and one that hard-codes 1/d^2 gets small ones wrong.
__host__ __device__ __forceinline__ float sample_power(
    float weight, float reflectance, float cos_theta, float range)
{
    return weight * reflectance * cos_theta / (range * range);
}

/// Add one return of amplitude `amp` arriving at time `t_r` into a waveform.
///
/// The Gaussian is INTEGRATED over each bin rather than sampled at bin centres, so a
/// return carries the same total energy wherever it falls between two bin edges. Point
/// sampling makes that energy wobble by a few percent with sub-bin phase, which looks
/// exactly like tolerance and is in fact a bug -- it sinks the 1/d^2 invariant for reasons
/// having nothing to do with range.
///
/// Shared by the direct and the multi-bounce transport models on purpose. Two copies of an
/// energy-conserving bin integral is precisely the duplication that let the baseline
/// tracer and the physics quietly stop describing the same geometry in Phase 2.
__device__ __forceinline__ void splat_pulse(
    float* wave, const SensorConfig& cfg, float t_r, float amp, float sigma)
{
    const float inv_sig_sqrt2 = 1.0f / (sigma * 1.41421356f);
    const float t_lo = t_r - 4.0f * sigma, t_hi = t_r + 4.0f * sigma;
    int k0 = static_cast<int>(floorf((t_lo - cfg.t0_seconds) / cfg.bin_seconds));
    int k1 = static_cast<int>(ceilf ((t_hi - cfg.t0_seconds) / cfg.bin_seconds));
    k0 = max(k0, 0); k1 = min(k1, cfg.bins - 1);
    for (int k = k0; k <= k1; ++k) {
        const float e0 = cfg.t0_seconds + k * cfg.bin_seconds;
        const float e1 = e0 + cfg.bin_seconds;
        const float frac = 0.5f * (erff((e1 - t_r) * inv_sig_sqrt2)
                                 - erff((e0 - t_r) * inv_sig_sqrt2));
        if (frac > 0.0f) atomicAdd(&wave[k], amp * frac);
    }
}

// -----------------------------------------------------------------------------
// The kernel
// -----------------------------------------------------------------------------

namespace detail {

/// Shared-memory ground-truth table: open addressing on a quantised range key.
constexpr int kTruthSlots = 64;
/// Truth is clustered four times finer than the merge distance, then neighbours are
/// merged. Quantising straight to the merge distance would split any surface unlucky
/// enough to straddle a cell boundary into two returns, which is an artifact of the grid
/// and not of the scene.
constexpr float kTruthSubdiv = 4.0f;

}  // namespace detail

/// Trace one beam per block and record both the waveform and what produced it.
///
/// Shared memory layout, allocated dynamically by the caller:
///   [0,    bins)        waveform accumulator
///   [bins, bins + 1)    sum of sample weights
///   then the truth table: keys, w, wr, wc, material, sample count
__global__ void lidar_trace(
    const BvhNode* __restrict__ nodes,
    const int* __restrict__ indices,
    const Triangle* __restrict__ tris,
    const Material* __restrict__ mats,
    const Beam* __restrict__ beams,
    SensorConfig cfg,
    float* __restrict__ waveforms,      // n_beams * cfg.bins
    TruthReturn* __restrict__ truth,    // n_beams * cfg.max_returns
    int* __restrict__ n_returns,        // n_beams
    int n_beams)
{
    const int beam_id = blockIdx.x;
    if (beam_id >= n_beams) return;

    extern __shared__ unsigned char smem_raw[];
    float* wave   = reinterpret_cast<float*>(smem_raw);
    float* wsum   = wave + cfg.bins;
    int*   t_key  = reinterpret_cast<int*>(wsum + 1);
    float* t_w    = reinterpret_cast<float*>(t_key + detail::kTruthSlots);
    float* t_wr   = t_w  + detail::kTruthSlots;
    float* t_wc   = t_wr + detail::kTruthSlots;
    int*   t_mat  = reinterpret_cast<int*>(t_wc + detail::kTruthSlots);
    int*   t_n    = t_mat + detail::kTruthSlots;

    for (int k = threadIdx.x; k < cfg.bins; k += blockDim.x) wave[k] = 0.0f;
    for (int k = threadIdx.x; k < detail::kTruthSlots; k += blockDim.x) {
        t_key[k] = -1; t_w[k] = 0.0f; t_wr[k] = 0.0f; t_wc[k] = 0.0f;
        t_mat[k] = -1; t_n[k] = 0;
    }
    if (threadIdx.x == 0) *wsum = 0.0f;
    __syncthreads();

    const Beam beam = beams[beam_id];
    const float sigma = pulse_sigma(cfg);
    const float inv_half_c = 2.0f / static_cast<float>(kSpeedOfLight);
    const float truth_cell = cfg.truth_merge_m / detail::kTruthSubdiv;

    float w_local = 0.0f;

    for (int s = threadIdx.x; s < cfg.samples(); s += blockDim.x) {
        float3 dir; float w;
        sample_beam(beam, cfg, beam_id, s, dir, w);
        w_local += w;

        Ray r;
        r.origin = beam.origin;
        r.direction = dir;
        r.tmax = cfg.max_range();
        const Hit h = traverse_bvh(nodes, indices, tris, r);
        if (h.t <= 0.0f) continue;

        const float d = h.t;
        const float cos_theta = fabsf(d_dot(h.normal, dir));
        const float rho = mats[h.material].reflectance;
        const float amp = sample_power(w, rho, cos_theta, d);

        const float t_r = d * inv_half_c;              // round trip: 2d/c
        splat_pulse(wave, cfg, t_r, amp, sigma);

        // --- record the truth ------------------------------------------------
        const int key = static_cast<int>(floorf(d / truth_cell));
        const int slot = static_cast<int>(hash_u32(static_cast<unsigned>(key))
                                          % detail::kTruthSlots);
        for (int probe = 0; probe < detail::kTruthSlots; ++probe) {
            const int idx = (slot + probe) % detail::kTruthSlots;
            const int prev = atomicCAS(&t_key[idx], -1, key);
            if (prev == -1 || prev == key) {
                atomicAdd(&t_w[idx], w);
                atomicAdd(&t_wr[idx], w * d);
                atomicAdd(&t_wc[idx], w * cos_theta);
                atomicAdd(&t_n[idx], 1);
                t_mat[idx] = h.material;
                break;
            }
        }
    }

    atomicAdd(wsum, w_local);
    __syncthreads();

    // Normalise by the total emitted weight so a waveform is in units independent of the
    // sample count -- otherwise "more samples" would read as "more energy" and the
    // convergence check would be measuring the wrong thing.
    const float inv_w = (*wsum > 0.0f) ? (1.0f / *wsum) : 0.0f;
    for (int k = threadIdx.x; k < cfg.bins; k += blockDim.x)
        waveforms[static_cast<size_t>(beam_id) * cfg.bins + k] = wave[k] * inv_w;

    // --- collapse the truth table ---------------------------------------------
    // Serial in one thread, over at most 64 slots. It is not the hot path, and a parallel
    // sort here would buy nothing but a place for an ordering bug to hide.
    if (threadIdx.x == 0) {
        TruthReturn tmp[detail::kTruthSlots];
        int m = 0;
        for (int i = 0; i < detail::kTruthSlots; ++i) {
            if (t_key[i] < 0 || t_w[i] <= 0.0f) continue;
            TruthReturn tr;
            tr.range     = t_wr[i] / t_w[i];
            tr.cos_theta = t_wc[i] / t_w[i];
            tr.weight    = t_w[i] * inv_w;
            tr.material  = t_mat[i];
            tr.samples   = t_n[i];
            tmp[m++] = tr;
        }
        for (int i = 1; i < m; ++i) {            // insertion sort by range
            const TruthReturn v = tmp[i];
            int j = i - 1;
            while (j >= 0 && tmp[j].range > v.range) { tmp[j + 1] = tmp[j]; --j; }
            tmp[j + 1] = v;
        }
        int out = 0;                             // merge what the grid split apart
        for (int i = 0; i < m; ++i) {
            if (out > 0 && tmp[i].range - tmp[out - 1].range < cfg.truth_merge_m
                        && tmp[i].material == tmp[out - 1].material) {
                TruthReturn& a = tmp[out - 1];
                const float wt = a.weight + tmp[i].weight;
                a.range     = (a.range * a.weight + tmp[i].range * tmp[i].weight) / wt;
                a.cos_theta = (a.cos_theta * a.weight + tmp[i].cos_theta * tmp[i].weight) / wt;
                a.weight    = wt;
                a.samples  += tmp[i].samples;
            } else {
                tmp[out++] = tmp[i];
            }
        }
        while (out > cfg.max_returns) {          // keep the strongest, still range-ordered
            int weakest = 0;
            for (int i = 1; i < out; ++i) if (tmp[i].weight < tmp[weakest].weight) weakest = i;
            for (int i = weakest; i < out - 1; ++i) tmp[i] = tmp[i + 1];
            --out;
        }
        n_returns[beam_id] = out;
        for (int i = 0; i < out; ++i)
            truth[static_cast<size_t>(beam_id) * cfg.max_returns + i] = tmp[i];
    }
}

/// Bytes of dynamic shared memory lidar_trace needs for this configuration.
inline size_t lidar_smem_bytes(const SensorConfig& cfg) {
    return sizeof(float) * (static_cast<size_t>(cfg.bins) + 1)
         + sizeof(int)   * detail::kTruthSlots * 3
         + sizeof(float) * detail::kTruthSlots * 3;
}

}  // namespace argos
