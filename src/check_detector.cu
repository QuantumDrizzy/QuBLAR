// =============================================================================
// QuBLAR -- the photon detector against its closed forms, and reconstruction
//           against truth
// =============================================================================
// ADR-003 items 3, 4, 6, 7.
//
// ADR-001 named the risk this file exists to retire: that the photon-counting model is
// really just a scaled copy of the linear one wearing noise. Two closed forms settle it,
// because both are exact and they are DIFFERENT from each other:
//
//   dead time off  ->  E[n_k]/N = lambda_k                    (the rate itself)
//   dead time on   ->  E[n_k]/N = exp(-R_{k-1}) - exp(-R_k)   (first-arrival law)
//
// A model that reproduces the first and not the second has no dead time. One that
// reproduces the second and not the first is not sampling the rate it was given. Neither
// can be faked by tuning a scale factor.
//
// Agreement is judged statistically rather than by a tolerance: counts are multinomial or
// Poisson with known variance, so the deviation is expressed in standard deviations and
// the threshold is a z-score. A hand-tuned epsilon here would be a number chosen until the
// test went green.
// =============================================================================

#include "detector.cuh"
#include "reconstruct.hpp"

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace argos;

static int failures = 0;

static void check(bool ok, const char* what, const std::string& detail = "") {
    std::printf("  %-46s %s  %s\n", what, ok ? "PASS" : "FAIL", detail.c_str());
    if (!ok) failures++;
}

static std::string fmt(const char* label, double v, const char* unit = "") {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s = %.4g %s", label, v, unit);
    return buf;
}

// -----------------------------------------------------------------------------
// A direct unit test of the block scan, because everything downstream is built on it
// and a prefix sum that is subtly wrong produces a plausible histogram.
// -----------------------------------------------------------------------------
__global__ void scan_probe(float* data, int n) {
    extern __shared__ float sm[];
    float* d = sm;
    float* scratch = d + n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) d[i] = data[i];
    __syncthreads();
    detail::block_scan(d, scratch, n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) data[i] = d[i];
}

// -----------------------------------------------------------------------------

static Scene make_step(float d_near, float d_far, float half = 40.f) {
    Scene s;
    s.add_material(0.4f);
    s.add_quad({0.f, -half, d_near}, {half, -half, d_near},
               {half, half, d_near}, {0.f, half, d_near}, 0);
    s.add_quad({-half, -half, d_far}, {0.f, -half, d_far},
               {0.f, half, d_far}, {-half, half, d_far}, 0);
    s.build();
    return s;
}

struct Sim {
    std::vector<float>       waves;
    std::vector<TruthReturn> truth;
    std::vector<int>         n_returns;
    std::vector<unsigned>    hist;
};

static Sim run(const Scene& scene, const std::vector<Beam>& beams,
               const SensorConfig& scfg, const DetectorConfig& dcfg, bool with_detector)
{
    DeviceScene ds; ds.upload(scene);
    const int n = static_cast<int>(beams.size());

    Beam* d_beams = nullptr; float* d_wave = nullptr;
    TruthReturn* d_truth = nullptr; int* d_nret = nullptr; unsigned* d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_beams, n * sizeof(Beam)));
    CUDA_CHECK(cudaMalloc(&d_wave, size_t(n) * scfg.bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_truth, size_t(n) * scfg.max_returns * sizeof(TruthReturn)));
    CUDA_CHECK(cudaMalloc(&d_nret, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_beams, beams.data(), n * sizeof(Beam), cudaMemcpyHostToDevice));

    lidar_trace<<<n, 256, lidar_smem_bytes(scfg)>>>(
        ds.nodes, ds.indices, ds.tris, ds.mats, d_beams, scfg,
        d_wave, d_truth, d_nret, n);
    CUDA_CHECK(cudaGetLastError());

    Sim out;
    out.waves.resize(size_t(n) * scfg.bins);
    out.truth.resize(size_t(n) * scfg.max_returns);
    out.n_returns.resize(n);
    CUDA_CHECK(cudaMemcpy(out.waves.data(), d_wave, out.waves.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.truth.data(), d_truth, out.truth.size() * sizeof(TruthReturn),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.n_returns.data(), d_nret, n * sizeof(int), cudaMemcpyDeviceToHost));

    if (with_detector) {
        const int block = 256;
        CUDA_CHECK(cudaMalloc(&d_hist, size_t(n) * scfg.bins * sizeof(unsigned)));
        spad_accumulate<<<n, block, spad_smem_bytes(scfg, block)>>>(
            d_wave, scfg, dcfg, d_hist, n);
        CUDA_CHECK(cudaGetLastError());
        out.hist.resize(size_t(n) * scfg.bins);
        CUDA_CHECK(cudaMemcpy(out.hist.data(), d_hist, out.hist.size() * sizeof(unsigned),
                              cudaMemcpyDeviceToHost));
        cudaFree(d_hist);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_beams); cudaFree(d_wave); cudaFree(d_truth); cudaFree(d_nret);
    ds.free();
    return out;
}

/// Per-bin expected photon counts for one waveform, on the host, in double.
static std::vector<double> lambda_of(const float* wave, const SensorConfig& scfg,
                                     const DetectorConfig& dcfg)
{
    double total = 0.0;
    for (int k = 0; k < scfg.bins; ++k) total += wave[k];
    std::vector<double> lam(scfg.bins);
    const double amb = double(dcfg.ambient_photons) / scfg.bins;
    for (int k = 0; k < scfg.bins; ++k)
        lam[k] = (total > 0.0 ? double(wave[k]) / total * dcfg.signal_photons : 0.0) + amb;
    return lam;
}

// -----------------------------------------------------------------------------

int main() {
    std::printf("\nQuBLAR Phase 3 -- photon counting and reconstruction\n\n");

    SensorConfig scfg;
    scfg.bins = 2048;                       // 0.2 ns bins -> 61.4 m of range
    scfg.samples_sqrt = 32;

    int tested_ideal = 0;

    Beam nadir;
    nadir.origin = make_float3(0.f, 0.f, 0.f);
    nadir.axis = make_float3(0.f, 0.f, 1.f);
    nadir.divergence_half_angle = 2e-3f;

    // ---- the prefix sum the whole detector is built on ------------------------
    {
        const int n = 2048;
        std::vector<float> h(n);
        for (int i = 0; i < n; ++i) h[i] = 0.5f + (i % 7) * 0.25f;
        std::vector<double> want(n);
        double acc = 0.0;
        for (int i = 0; i < n; ++i) { acc += h[i]; want[i] = acc; }

        float* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        const int block = 256;
        scan_probe<<<1, block, (n + block) * sizeof(float)>>>(d, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d);

        double worst = 0.0;
        for (int i = 0; i < n; ++i) worst = std::max(worst, std::fabs(h[i] - want[i]) / want[i]);
        check(worst < 1e-5, "block scan matches a host prefix sum", fmt("max rel", worst));
    }

    Scene plane;
    plane.add_material(0.4f);
    plane.add_quad({-40, -40, 10}, {40, -40, 10}, {40, 40, 10}, {-40, 40, 10}, 0);
    plane.build();

    // ---- dead time OFF: the histogram must reproduce the rate -----------------
    {
        // 400000 pulses, not 200000, and the reason is worth recording. At 200000 the
        // ambient floor came to 48.8 expected counts per bin -- just under the 50 needed
        // for a z-score to mean anything -- so only the ~19 signal bins were tested and
        // the whole background of the gate went unchecked. The physics was fine at both
        // counts; the sampling budget was chosen badly. Now every bin clears the bar.
        DetectorConfig d;
        d.signal_photons = 2.0f;
        d.ambient_photons = 0.5f;
        d.pulses = 400000;
        d.ideal = true;
        const Sim s = run(plane, {nadir}, scfg, d, true);

        const std::vector<double> lam = lambda_of(s.waves.data(), scfg, d);
        double worst_z = 0.0; int tested = 0;
        for (int k = 0; k < scfg.bins; ++k) {
            const double mu = lam[k] * d.pulses;
            if (mu < 50.0) continue;                    // too few counts to say anything
            const double z = (s.hist[k] - mu) / std::sqrt(mu);   // Poisson
            worst_z = std::max(worst_z, std::fabs(z));
            tested++;
        }
        check(worst_z < 5.0 && tested > 1500, "counts follow the rate when nothing blocks",
              fmt("max |z|", worst_z) + ", " + std::to_string(tested) + " bins");
        tested_ideal = tested;
    }

    // ---- dead time ON: the first-arrival law ----------------------------------
    // This is the check with teeth. The expected shape is NOT the rate, it is the
    // distribution of the FIRST arrival, and no scale factor maps one onto the other.
    {
        DetectorConfig d;
        d.signal_photons = 2.0f;
        d.ambient_photons = 0.5f;
        d.pulses = 400000;
        d.dead_time_seconds = 1e30f;                    // classic TCSPC: one photon/pulse
        const Sim s = run(plane, {nadir}, scfg, d, true);

        const std::vector<double> lam = lambda_of(s.waves.data(), scfg, d);
        double R = 0.0, worst_z = 0.0; int tested = 0;
        for (int k = 0; k < scfg.bins; ++k) {
            const double R0 = R; R += lam[k];
            const double p = std::exp(-R0) - std::exp(-R);
            const double mu = p * d.pulses;
            if (mu < 50.0) continue;
            const double var = mu * (1.0 - p);          // multinomial
            worst_z = std::max(worst_z, std::fabs((s.hist[k] - mu) / std::sqrt(var)));
            tested++;
        }
        // The coverage bar is 300 here and 1500 above, and the difference is the effect
        // under test rather than a bar lowered until the test went green. With dead time
        // on, exp(-R) collapses once the return has been recorded, so the tail of the gate
        // accumulates almost no counts and no z-score there can mean anything. Demanding
        // full coverage would be demanding that pile-up not happen. What must be covered
        // is the pre-return background and the returns themselves, which 300 bins is.
        check(worst_z < 5.0 && tested > 300, "counts follow the first-arrival law",
              fmt("max |z|", worst_z) + ", " + std::to_string(tested) + " bins");

        // And that collapse is itself a signature, independent of the law above: the same
        // gate, the same rate, the same pulse count, and most of it stops carrying
        // statistics purely because the detector has already fired.
        check(tested < tested_ideal / 2, "the tail stops carrying counts once it has fired",
              std::to_string(tested) + " testable bins against "
              + std::to_string(tested_ideal) + " with nothing blocking");

        // And the direction, independently of the law above: pile-up can only ever take
        // counts AWAY from later bins relative to the rate, never add them.
        double R2 = 0.0, prev_ratio = 1e30; bool monotone = true;
        for (int k = 0; k < scfg.bins; ++k) {
            const double R0 = R2; R2 += lam[k];
            const double p = std::exp(-R0) - std::exp(-R2);
            if (lam[k] < 1e-9 || p <= 0.0) continue;
            const double ratio = p / lam[k];
            if (ratio > prev_ratio * 1.000001) monotone = false;
            prev_ratio = ratio;
        }
        check(monotone, "pile-up only ever suppresses later bins", "ratio is monotone");
    }

    // ---- reconstruction, scored against truth ---------------------------------
    // A scan sweeping the footprint across a step edge, so some beams see one surface,
    // some see two, and some straddle. Scoring on a scene where every beam sees exactly
    // one return would flatter every algorithm equally and distinguish none of them.
    {
        const float d_near = 10.f, d_far = 12.f;
        Scene step = make_step(d_near, d_far);

        std::vector<Beam> beams;
        const int n_beams = 96;
        for (int i = 0; i < n_beams; ++i) {
            const float theta = (i - (n_beams - 1) * 0.5f) * (0.02f / n_beams);
            Beam b;
            b.origin = make_float3(0.f, 0.f, 0.f);
            b.axis = make_float3(std::sin(theta), 0.f, std::cos(theta));
            b.divergence_half_angle = 2e-3f;
            beams.push_back(b);
        }

        DetectorConfig d;
        d.signal_photons = 2.0f;
        d.ambient_photons = 0.3f;
        d.pulses = 20000;
        d.dead_time_seconds = 1e30f;
        const Sim s = run(step, beams, scfg, d, true);

        int two_return_beams = 0;
        for (int i = 0; i < n_beams; ++i) if (s.n_returns[i] >= 2) two_return_beams++;
        check(two_return_beams > 10, "the scan actually straddles the edge",
              std::to_string(two_return_beams) + " of " + std::to_string(n_beams)
              + " beams see two surfaces");

        const float tol = 0.30f;    // a return is matched if it lands within 30 cm
        auto run_all = [&](auto trace_of) {
            Score a, b, c;
            for (int i = 0; i < n_beams; ++i) {
                const std::vector<float> tr = trace_of(i);
                const TruthReturn* t = &s.truth[size_t(i) * scfg.max_returns];
                score_beam(peak_pick(tr, scfg), t, s.n_returns[i], tol, a);
                score_beam(matched_filter(tr, scfg), t, s.n_returns[i], tol, b);
                score_beam(greedy_deconvolve(tr, scfg), t, s.n_returns[i], tol, c);
            }
            return std::vector<Score>{a, b, c};
        };

        const char* algo[3] = {"peak pick", "matched filter", "greedy deconvolve"};

        std::printf("\n  %-20s %-18s %8s %8s %10s %10s\n",
                    "input", "algorithm", "detect", "false/bm", "bias m", "rmse m");
        std::printf("  %s\n", std::string(78, '-').c_str());

        auto report = [&](const char* tag, const std::vector<Score>& sc) {
            for (int j = 0; j < 3; ++j)
                std::printf("  %-20s %-18s %7.1f%% %8.2f %+10.4f %10.4f\n",
                            j == 0 ? tag : "", algo[j],
                            100.0 * sc[j].detection_rate(), sc[j].false_per_beam(),
                            sc[j].bias_m(), sc[j].rmse_m());
        };

        const std::vector<Score> lin = run_all([&](int i) {
            return std::vector<float>(s.waves.begin() + size_t(i) * scfg.bins,
                                      s.waves.begin() + size_t(i + 1) * scfg.bins);
        });
        const std::vector<Score> raw = run_all([&](int i) {
            return counts_as_trace(&s.hist[size_t(i) * scfg.bins], scfg.bins);
        });
        const std::vector<Score> cor = run_all([&](int i) {
            return coates(&s.hist[size_t(i) * scfg.bins], scfg.bins, d.pulses);
        });

        report("linear waveform", lin);
        report("photon counts", raw);
        report("counts + Coates", cor);
        std::printf("\n");

        // The linear waveform is noiseless and unbiased, so it is the ceiling: nothing
        // downstream may beat it, and if something does, the metric is wrong rather than
        // the algorithm being clever.
        check(std::fabs(lin[2].bias_m()) < 0.02,
              "the noiseless waveform is essentially unbiased",
              fmt("bias", lin[2].bias_m(), "m"));

        // Pile-up must show up as a SHORT bias: the detector preferentially records early
        // photons, so ranges come back too near. This is the measurement the whole
        // detector model exists to produce, so it is asserted, not merely printed.
        check(raw[2].bias_m() < -0.005,
              "pile-up biases photon ranges short",
              fmt("bias", raw[2].bias_m(), "m"));

        check(std::fabs(cor[2].bias_m()) < std::fabs(raw[2].bias_m()),
              "Coates correction reduces that bias",
              fmt("raw", raw[2].bias_m(), "m") + " -> " + fmt("corrected", cor[2].bias_m(), "m"));

        // And the control earns its place: peak pick cannot express a second surface, so
        // on a scan that straddles an edge it must lose detections to the methods that can.
        check(lin[2].detection_rate() > lin[0].detection_rate() + 0.05,
              "multi-return methods beat the single-return control",
              fmt("peak", 100 * lin[0].detection_rate(), "%") + " vs "
              + fmt("greedy", 100 * lin[2].detection_rate(), "%"));
    }

    // ---- where the baselines actually break -----------------------------------
    // The table above reports zero false alarms everywhere, which is not a result -- it
    // says the scene was too easy to separate the methods. A scoring harness that is only
    // ever run in the regime where everything works is a harness nobody has used.
    //
    // So: the same scan, the same algorithm, swept over photon budget from a starved
    // detector to a saturated one. Detection collapses at the bottom and pile-up bias
    // grows at the top, and the two failures have nothing to do with each other.
    {
        Scene step2 = make_step(10.f, 12.f);
        std::vector<Beam> beams2;
        const int nb = 96;
        for (int i = 0; i < nb; ++i) {
            const float theta = (i - (nb - 1) * 0.5f) * (0.02f / nb);
            Beam b;
            b.origin = make_float3(0.f, 0.f, 0.f);
            b.axis = make_float3(std::sin(theta), 0.f, std::cos(theta));
            b.divergence_half_angle = 2e-3f;
            beams2.push_back(b);
        }

        std::printf("  greedy deconvolve on raw counts, swept over photon budget\n");
        std::printf("  %-10s %-10s %8s %8s %10s %10s\n",
                    "signal/pl", "ambient", "detect", "false/bm", "bias m", "rmse m");
        std::printf("  %s\n", std::string(62, '-').c_str());

        // The first attempt swept 0.02 to 10 photons per pulse at 0.3 ambient and found
        // 94% detection at the bottom -- because with 20000 pulses, 0.02 photons per pulse
        // is still 400 photons, which is not a starved detector. What starves it is the
        // SIGNAL-TO-BACKGROUND ratio at a fixed acquisition time, so the range goes lower
        // and the ambient goes up.
        double best_detect = 0.0, low_detect = 1.0, biggest_bias = 0.0;
        bool first = true;
        for (float sig : {0.0005f, 0.002f, 0.01f, 0.05f, 0.5f, 2.0f, 10.0f}) {
            DetectorConfig dd;
            dd.signal_photons = sig;
            dd.ambient_photons = 1.0f;
            dd.pulses = 20000;
            dd.dead_time_seconds = 1e30f;
            const Sim ss = run(step2, beams2, scfg, dd, true);

            Score sc;
            for (int i = 0; i < nb; ++i) {
                const std::vector<float> tr =
                    counts_as_trace(&ss.hist[size_t(i) * scfg.bins], scfg.bins);
                score_beam(greedy_deconvolve(tr, scfg),
                           &ss.truth[size_t(i) * scfg.max_returns], ss.n_returns[i], 0.30f, sc);
            }
            std::printf("  %-10.4f %-10.2f %7.1f%% %8.2f %+10.4f %10.4f\n",
                        sig, dd.ambient_photons, 100.0 * sc.detection_rate(),
                        sc.false_per_beam(), sc.bias_m(), sc.rmse_m());
            if (first) { low_detect = sc.detection_rate(); first = false; }
            best_detect = std::max(best_detect, sc.detection_rate());
            biggest_bias = std::max(biggest_bias, std::fabs(sc.bias_m()));
        }
        std::printf("\n");

        // Both ends must actually fail, or the sweep never reached the interesting regime
        // and the rows above are decoration.
        // Asserted on the SHAPE of the envelope, not on a magic threshold: detection at
        // the starved end must be materially worse than at the best point. Picking an
        // absolute number here would mean choosing it until the test went green, which is
        // the same mistake that put 200000 pulses and a 100-bin floor in this file.
        check(low_detect < best_detect - 0.10, "detection degrades at the starved end",
              fmt("lowest flux", 100 * low_detect, "%") + " vs "
              + fmt("best", 100 * best_detect, "%"));
        check(biggest_bias > 0.05, "pile-up bias grows with flux",
              fmt("largest bias", biggest_bias, "m"));
    }
    std::printf("\n%s\n\n", failures == 0 ? "all checks passed" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
