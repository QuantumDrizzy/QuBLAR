// =============================================================================
// QuBLAR -- reconstruction baselines, and scoring them against truth
// =============================================================================
// ADR-003 items 5-7. Three ways of turning a waveform or a photon histogram back into
// a list of surfaces, chosen so that their failure modes differ rather than so that one
// of them wins.
//
// Host code on purpose. These are per-waveform and cheap, they are the thing the
// simulator is scored WITH rather than part of it, and their correctness matters far
// more than their speed. Putting them on the GPU would buy nothing and would put the
// yardstick in the same place as the thing being measured.
// =============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "lidar.cuh"

namespace argos {

/// One surface an algorithm believes it found.
struct Estimate {
    float range;       // metres
    float strength;    // in whatever units the input carried; only ordering is used
};

// -----------------------------------------------------------------------------
// Inputs
// -----------------------------------------------------------------------------

/// Turn a photon histogram into an intensity trace an algorithm can work on.
/// Raw counts, no correction. This is the honest starting point: it is what the
/// instrument hands you.
inline std::vector<float> counts_as_trace(const unsigned* hist, int bins) {
    std::vector<float> v(bins);
    for (int k = 0; k < bins; ++k) v[k] = static_cast<float>(hist[k]);
    return v;
}

/// Coates correction: the maximum-likelihood estimate of the true per-bin rate from a
/// pile-up-distorted histogram.
///
///     lambda_k = ln( (N - sum_{j<k} n_j) / (N - sum_{j<=k} n_j) )
///
/// The reasoning is exactly the physical one: of the N pulses, only those that had not
/// already fired before bin k were ever able to record a photon IN bin k. Dividing by
/// that surviving population removes the bias.
///
/// It is offered as its own reconstruction path rather than applied to everything, so the
/// scoring table can show the same algorithm with and without it. Correcting silently
/// would hide the effect the detector model exists to produce.
inline std::vector<float> coates(const unsigned* hist, int bins, int pulses) {
    std::vector<float> v(bins, 0.f);
    double before = 0.0;
    for (int k = 0; k < bins; ++k) {
        const double survivors = pulses - before;
        const double after = before + hist[k];
        const double remaining = pulses - after;
        // The estimator diverges once every pulse has fired. That is not a numerical
        // problem to be clamped away quietly -- it means the bin carries no information,
        // so it is reported as zero and the saturation is visible as a flat tail.
        v[k] = (survivors > 0.0 && remaining > 0.0)
             ? static_cast<float>(std::log(survivors / remaining))
             : 0.0f;
        before = after;
    }
    return v;
}

// -----------------------------------------------------------------------------
// Baselines
// -----------------------------------------------------------------------------

inline float bin_to_range(int k, const SensorConfig& cfg) {
    const double t = cfg.t0_seconds + (k + 0.5) * cfg.bin_seconds;
    return static_cast<float>(0.5 * t * kSpeedOfLight);
}

/// The control. One return, at the strongest bin.
///
/// It cannot express a second surface at all, which is the point of including it: if a
/// more elaborate method does not beat it on the multi-return scenes, the elaboration is
/// not paying for itself.
inline std::vector<Estimate> peak_pick(const std::vector<float>& trace,
                                       const SensorConfig& cfg) {
    int best = 0;
    for (size_t k = 1; k < trace.size(); ++k) if (trace[k] > trace[best]) best = static_cast<int>(k);
    if (trace[best] <= 0.f) return {};
    return {{bin_to_range(best, cfg), trace[best]}};
}

/// Normalised Gaussian matching the transmitted pulse, sampled on the bin grid.
inline std::vector<float> pulse_kernel(const SensorConfig& cfg, int& half) {
    const double sigma = pulse_sigma(cfg);
    half = std::max(1, static_cast<int>(std::ceil(3.0 * sigma / cfg.bin_seconds)));
    std::vector<float> k(2 * half + 1);
    double norm = 0.0;
    for (int i = -half; i <= half; ++i) {
        const double x = i * cfg.bin_seconds / sigma;
        const double v = std::exp(-0.5 * x * x);
        k[i + half] = static_cast<float>(v);
        norm += v * v;
    }
    const double inv = 1.0 / std::sqrt(norm);
    for (auto& v : k) v = static_cast<float>(v * inv);
    return k;
}

inline std::vector<float> correlate(const std::vector<float>& trace,
                                    const std::vector<float>& kern, int half) {
    const int n = static_cast<int>(trace.size());
    std::vector<float> out(n, 0.f);
    for (int k = 0; k < n; ++k) {
        double acc = 0.0;
        for (int i = -half; i <= half; ++i) {
            const int j = k + i;
            if (j >= 0 && j < n) acc += double(trace[j]) * kern[i + half];
        }
        out[k] = static_cast<float>(acc);
    }
    return out;
}

/// Correlate with the known pulse, keep local maxima above a threshold.
///
/// The threshold is expressed in units of the trace's own noise floor rather than as an
/// absolute, because the same absolute number means different things at 0.1 and 5 photons
/// per pulse, and a baseline that needs retuning per scene is not a baseline.
inline std::vector<Estimate> matched_filter(const std::vector<float>& trace,
                                            const SensorConfig& cfg,
                                            float threshold_sigmas = 5.0f,
                                            float min_fraction = 0.05f) {
    int half = 0;
    const std::vector<float> kern = pulse_kernel(cfg, half);
    const std::vector<float> resp = correlate(trace, kern, half);

    // Median absolute deviation, not the standard deviation: the returns themselves are
    // large outliers, so an ordinary sigma is inflated by the very signal being detected
    // and the threshold would scale with the thing it is supposed to be independent of.
    std::vector<float> sorted = resp;
    std::sort(sorted.begin(), sorted.end());
    const float med = sorted[sorted.size() / 2];
    for (auto& v : sorted) v = std::fabs(v - med);
    std::sort(sorted.begin(), sorted.end());
    const float mad = sorted[sorted.size() / 2];
    const float sigma = std::max(1.4826f * mad, 1e-12f);

    // A floor relative to the strongest response, as well as the noise threshold. On a
    // noiseless trace the median absolute deviation is zero, the noise threshold collapses
    // to nothing, and every floating-point ripple becomes a detected surface. Declaring a
    // return below 5% of the peak to be no return is standard waveform practice and is
    // stated rather than buried, because it sets the weakest surface this can ever find.
    float peak = 0.f;
    for (float v : resp) peak = std::max(peak, v);
    const float cut = std::max(med + threshold_sigmas * sigma, min_fraction * peak);

    std::vector<Estimate> out;
    const int n = static_cast<int>(resp.size());
    for (int k = 1; k < n - 1; ++k)
        if (resp[k] > cut && resp[k] >= resp[k - 1] && resp[k] > resp[k + 1])
            out.push_back({bin_to_range(k, cfg), resp[k]});

    std::sort(out.begin(), out.end(),
              [](const Estimate& a, const Estimate& b) { return a.range < b.range; });
    return out;
}

/// Greedy deconvolution: find the strongest matched-filter peak, subtract the pulse that
/// would have produced it, repeat.
///
/// Resolves closer pairs than plain peak detection, because the first return is removed
/// before the second is looked for. Its characteristic failure is the mirror of that
/// strength: error in the first fit is subtracted into the residual and shows up as a
/// spurious later return, so the iteration count and the floor both matter and are named
/// rather than hidden.
inline std::vector<Estimate> greedy_deconvolve(const std::vector<float>& trace,
                                               const SensorConfig& cfg,
                                               int max_returns = 4,
                                               float threshold_sigmas = 5.0f,
                                               float min_fraction = 0.05f) {
    int half = 0;
    const std::vector<float> kern = pulse_kernel(cfg, half);
    std::vector<float> residual = trace;

    std::vector<float> resp0 = correlate(trace, kern, half);
    std::vector<float> sorted = resp0;
    std::sort(sorted.begin(), sorted.end());
    const float med = sorted[sorted.size() / 2];
    for (auto& v : sorted) v = std::fabs(v - med);
    std::sort(sorted.begin(), sorted.end());
    const float sigma = std::max(1.4826f * sorted[sorted.size() / 2], 1e-12f);
    float peak0 = 0.f;
    for (float v : resp0) peak0 = std::max(peak0, v);
    const float cut = std::max(med + threshold_sigmas * sigma, min_fraction * peak0);

    const double psigma = pulse_sigma(cfg);
    std::vector<Estimate> out;

    for (int it = 0; it < max_returns; ++it) {
        const std::vector<float> resp = correlate(residual, kern, half);
        int best = 0;
        for (size_t k = 1; k < resp.size(); ++k) if (resp[k] > resp[best]) best = static_cast<int>(k);
        if (resp[best] <= cut) break;

        // Sub-bin position by a parabola through the peak and its neighbours. Without it
        // every estimate is quantised to the bin grid, which at 0.2 ns is 3 cm -- larger
        // than the range error the better algorithms actually achieve, so the metric would
        // be measuring the grid rather than the method.
        float shift = 0.f;
        if (best > 0 && best + 1 < static_cast<int>(resp.size())) {
            const float a = resp[best - 1], b = resp[best], c = resp[best + 1];
            const float den = a - 2 * b + c;
            if (std::fabs(den) > 1e-20f) shift = 0.5f * (a - c) / den;
        }
        const double t = cfg.t0_seconds + (best + 0.5 + shift) * cfg.bin_seconds;
        out.push_back({static_cast<float>(0.5 * t * kSpeedOfLight), residual[best]});

        const float amp = residual[best];
        for (int i = -half; i <= half; ++i) {
            const int j = best + i;
            if (j < 0 || j >= static_cast<int>(residual.size())) continue;
            const double x = (i - shift) * cfg.bin_seconds / psigma;
            residual[j] -= static_cast<float>(amp * std::exp(-0.5 * x * x));
        }
    }
    std::sort(out.begin(), out.end(),
              [](const Estimate& a, const Estimate& b) { return a.range < b.range; });
    return out;
}

// -----------------------------------------------------------------------------
// Scoring
// -----------------------------------------------------------------------------

struct Score {
    int    truth_returns = 0;
    int    matched = 0;
    int    false_alarms = 0;
    int    beams = 0;
    double sum_err = 0.0;      // signed, so bias survives
    double sum_sq = 0.0;

    double detection_rate() const { return truth_returns ? double(matched) / truth_returns : 0.0; }
    double false_per_beam() const { return beams ? double(false_alarms) / beams : 0.0; }
    double bias_m() const { return matched ? sum_err / matched : 0.0; }
    double rmse_m() const { return matched ? std::sqrt(sum_sq / matched) : 0.0; }
};

/// Match estimates to truth, nearest pair first, one-to-one, within a tolerance.
///
/// Greedy-nearest rather than a full optimal assignment. With at most a handful of returns
/// per beam the two agree except in contrived cases, and the greedy version is short
/// enough to be read and believed -- which matters more here, since this function decides
/// who wins.
///
/// Detection, false alarms, bias and RMSE are accumulated separately and are never
/// combined. An algorithm can buy a perfect detection rate by reporting a return in every
/// bin, and a perfect false-alarm rate by reporting nothing at all; a single blended score
/// would conceal which of the two it is doing, and the weights would be making the
/// argument instead of the data.
inline void score_beam(const std::vector<Estimate>& est,
                       const TruthReturn* truth, int n_truth,
                       float tolerance_m, Score& s)
{
    s.beams++;
    s.truth_returns += n_truth;

    std::vector<char> used_e(est.size(), 0), used_t(n_truth, 0);
    while (true) {
        double best = tolerance_m;
        int bi = -1, bj = -1;
        for (size_t i = 0; i < est.size(); ++i) {
            if (used_e[i]) continue;
            for (int j = 0; j < n_truth; ++j) {
                if (used_t[j]) continue;
                const double d = std::fabs(double(est[i].range) - double(truth[j].range));
                if (d < best) { best = d; bi = static_cast<int>(i); bj = j; }
            }
        }
        if (bi < 0) break;
        used_e[bi] = 1; used_t[bj] = 1;
        const double err = double(est[bi].range) - double(truth[bj].range);
        s.matched++;
        s.sum_err += err;
        s.sum_sq  += err * err;
    }
    for (size_t i = 0; i < est.size(); ++i) if (!used_e[i]) s.false_alarms++;
}

}  // namespace argos
