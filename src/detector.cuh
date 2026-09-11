// =============================================================================
// QuBLAR -- the photon-counting detector (SPAD / TCSPC)
// =============================================================================
// ADR-003 items 1-2. Turns a linear-mode waveform into what a single-photon detector
// would actually record: a histogram of photon arrival times accumulated over many
// laser pulses, with a dead time that makes early photons suppress later ones.
//
// That suppression is the entire reason this file exists. A SPAD that fires is blind
// for tens of nanoseconds afterwards, so the histogram leans toward short ranges --
// PILE-UP -- and any algorithm validated on a simulator without it has been validated
// on data no instrument could produce.
//
// Arrivals are sampled by inverting the cumulative rate rather than by drawing a
// Poisson count in every bin. The per-bin version is slower and, more importantly,
// models a detector with no memory, which is exactly the property that makes single
// photon LiDAR difficult.
// =============================================================================

#pragma once

#include "lidar.cuh"

namespace argos {

struct DetectorConfig {
    /// Expected DETECTED signal photons per pulse, spread over the gate in proportion to
    /// the linear-mode waveform. Stated in photons rather than watts because an optical
    /// power budget for a specific instrument is not something this project can calibrate
    /// honestly, and a fabricated one would look more authoritative than it is.
    float signal_photons = 0.5f;

    /// Expected background photons per gate, uniform in time. This is daylight, and it is
    /// the reason single-photon LiDAR is hard outdoors rather than on a bench.
    float ambient_photons = 0.1f;

    /// Non-paralysable: a photon arriving during the dead time is lost but does not
    /// extend it. That is what an actively quenched SPAD does. The paralysable model
    /// saturates to zero counts at high flux; a real device saturates to a finite rate.
    ///
    /// A dead time at or beyond the gate length reduces this to classic TCSPC, where at
    /// most one photon is recorded per pulse.
    float dead_time_seconds = 1e30f;

    int      pulses = 4096;
    unsigned seed = 0xA5A5A5A5u;

    /// Set to true to disable blocking entirely. Not a physical detector -- it exists so
    /// the histogram can be checked against the rate it was drawn from, which is the only
    /// way to tell a working photon model from a scaled copy of the linear one.
    bool ideal = false;
};

namespace detail {

/// Inclusive prefix sum over `n` values in shared memory, two levels.
///
/// The serial alternative -- thread 0 walking the array -- is correct, four lines, and
/// leaves 255 of 256 threads idle for 2048 iterations per beam. It is written out here
/// instead because embedding a knowingly bad design and calling it a simplification is
/// how a generator ends up unusable at dataset scale.
__device__ __forceinline__ void block_scan(float* data, float* chunk_sums, int n) {
    const int nthreads = blockDim.x;
    const int per = (n + nthreads - 1) / nthreads;
    const int lo = threadIdx.x * per;
    const int hi = min(lo + per, n);

    float sum = 0.0f;
    for (int i = lo; i < hi; ++i) { sum += data[i]; data[i] = sum; }   // local inclusive
    chunk_sums[threadIdx.x] = sum;
    __syncthreads();

    // One thread scans the chunk totals. There are blockDim.x of them, not n, so this is
    // a short serial pass rather than the long one it replaces.
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int i = 0; i < nthreads; ++i) { const float v = chunk_sums[i]; chunk_sums[i] = acc; acc += v; }
    }
    __syncthreads();

    const float offset = chunk_sums[threadIdx.x];
    for (int i = lo; i < hi; ++i) data[i] += offset;
    __syncthreads();
}

/// Smallest index k with cum[k] >= target, or n if none. Binary search, because the
/// alternative is a linear walk per photon and the whole point of inverting the
/// cumulative was to avoid touching every bin.
__device__ __forceinline__ int upper_index(const float* cum, int n, float target) {
    int lo = 0, hi = n;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (cum[mid] < target) lo = mid + 1; else hi = mid;
    }
    return lo;
}

}  // namespace detail

/// One block per beam. Reads the beam's linear-mode waveform, builds the arrival-rate
/// cumulative, and accumulates a photon histogram over `cfg.pulses` laser shots.
///
/// Shared memory layout:
///   [0,    bins)      the cumulative rate, in expected photons
///   [bins, 2*bins)    the histogram, as float for atomicAdd, converted on the way out
///   then blockDim.x floats of scratch for the scan
__global__ void spad_accumulate(
    const float* __restrict__ waveforms,    // n_beams * bins, from lidar_trace
    SensorConfig scfg,
    DetectorConfig dcfg,
    unsigned* __restrict__ histograms,      // n_beams * bins
    int n_beams)
{
    const int beam = blockIdx.x;
    if (beam >= n_beams) return;

    extern __shared__ float smem[];
    float* cum  = smem;
    float* hist = cum + scfg.bins;
    float* scratch = hist + scfg.bins;

    const float* wave = waveforms + static_cast<size_t>(beam) * scfg.bins;

    // Total returned energy, needed to turn a waveform shape into a photon count. Summed
    // by the same scan that produces the cumulative, one pass instead of two.
    for (int k = threadIdx.x; k < scfg.bins; k += blockDim.x) {
        cum[k] = wave[k];
        hist[k] = 0.0f;
    }
    __syncthreads();
    detail::block_scan(cum, scratch, scfg.bins);

    const float total_energy = cum[scfg.bins - 1];
    const float amb_per_bin = dcfg.ambient_photons / scfg.bins;

    // Every thread must have READ cum[bins-1] before any thread OVERWRITES it in the loop
    // below. Without this barrier the thread that owns the last bin can store its rescaled
    // value while another thread is still reading the same slot for total_energy, and that
    // thread then computes its share of the cumulative against a corrupted normalisation.
    //
    // racecheck found it; nothing else could have. The histogram stays entirely plausible
    // when it happens -- only the photon budget is quietly wrong, by an amount that varies
    // with scheduling.
    __syncthreads();

    // Rebuild the cumulative in PHOTONS: signal scaled to the requested photon budget,
    // plus a uniform ambient floor. Done from the already-scanned array, so the scan runs
    // once rather than once per component.
    const float scale = (total_energy > 0.0f) ? (dcfg.signal_photons / total_energy) : 0.0f;
    for (int k = threadIdx.x; k < scfg.bins; k += blockDim.x)
        cum[k] = cum[k] * scale + amb_per_bin * (k + 1);
    __syncthreads();

    const float gate = scfg.gate_end();
    const float dt = scfg.bin_seconds;
    const float total_rate = cum[scfg.bins - 1];

    for (int p = threadIdx.x; p < dcfg.pulses; p += blockDim.x) {
        float t_live = scfg.t0_seconds;    // the detector is armed from here
        float r_live = 0.0f;               // cumulative already "used up" before t_live
        unsigned key = hash_u32(static_cast<unsigned>(beam) * 0x9E3779B9u
                              ^ static_cast<unsigned>(p) * 0x85EBCA6Bu ^ dcfg.seed);

        for (int photon = 0; photon < 64; ++photon) {
            key = hash_u32(key + 0x9E3779B9u);
            const float u = (key >> 8) * (1.0f / 16777216.0f);

            // −ln(1 − u) is the waiting time of a unit-rate Poisson process. Mapping it
            // through the cumulative gives the next arrival of the inhomogeneous one.
            const float target = r_live - __logf(fmaxf(1.0f - u, 1e-30f));
            if (target >= total_rate) break;                 // no further arrival in gate

            const int k = detail::upper_index(cum, scfg.bins, target);
            if (k >= scfg.bins) break;

            // Interpolate inside the bin instead of snapping to its edge. The rate is
            // piecewise constant, so this is exact, and snapping would stamp the bin grid
            // onto the arrival times -- an artefact that would survive into every
            // histogram and look like real time quantisation.
            const float prev = (k > 0) ? cum[k - 1] : 0.0f;
            const float lam = cum[k] - prev;
            const float frac = (lam > 0.0f) ? (target - prev) / lam : 0.0f;
            const float t = scfg.t0_seconds + (k + fminf(frac, 0.999999f)) * dt;
            if (t >= gate) break;

            atomicAdd(&hist[k], 1.0f);

            if (dcfg.ideal) {
                // No blocking: the process continues from where it was. This is not a
                // detector, it is the control that lets the histogram be compared against
                // the rate that generated it.
                r_live = target;
                continue;
            }

            t_live = t + dcfg.dead_time_seconds;
            if (t_live >= gate) break;
            // Where the detector comes back to life, expressed in cumulative units.
            const int kd = min(static_cast<int>((t_live - scfg.t0_seconds) / dt), scfg.bins - 1);
            const float base = (kd > 0) ? cum[kd - 1] : 0.0f;
            const float lamd = cum[kd] - base;
            const float fr = (t_live - scfg.t0_seconds) / dt - kd;
            r_live = base + lamd * fr;
        }
    }
    __syncthreads();

    for (int k = threadIdx.x; k < scfg.bins; k += blockDim.x)
        histograms[static_cast<size_t>(beam) * scfg.bins + k] =
            static_cast<unsigned>(hist[k]);
}

inline size_t spad_smem_bytes(const SensorConfig& cfg, int block) {
    return sizeof(float) * (2 * static_cast<size_t>(cfg.bins) + block);
}

}  // namespace argos
