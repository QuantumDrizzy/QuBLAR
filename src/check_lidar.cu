// =============================================================================
// QuBLAR -- validating the sensor model against physics, not against itself
// =============================================================================
// ADR-001 item 7. This file is the reason anything downstream is worth reading: a
// simulator that is subtly wrong produces plausible data forever, and every algorithm
// validated on it is validated on a fiction.
//
// The checks are chosen so that each one FAILS for a different reason:
//
//   analytic plane    -- the closed-form answer. Catches the radiometry outright.
//   zero divergence   -- the waveform must collapse to the transmitted pulse at 2d/c.
//                        Catches a wrong geometry term that the total energy hides.
//   sub-bin phase     -- energy must not depend on where a return falls between bin
//                        edges. Catches a point-sampled IRF, which looks like tolerance.
//   1/d^2 sweep       -- range dependence, independent of the absolute scale.
//   cos(theta) sweep  -- incidence dependence, likewise.
//   half-plane        -- the beam centred on an edge must return exactly half its
//                        energy, by symmetry. Catches a biased cone sampler, and gives
//                        Monte Carlo convergence an analytic target rather than a
//                        high-sample-count copy of itself.
//   step edge         -- two surfaces in one footprint must appear as two returns at
//                        their two analytic ranges. This is the case the project exists
//                        for, so it is checked rather than assumed.
//
// Host-side anchors are computed in double. Comparing float against float means a shared
// mistake cancels and the check passes while measuring nothing.
// =============================================================================

#include "lidar.cuh"

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace argos;

// -----------------------------------------------------------------------------
// Reporting
// -----------------------------------------------------------------------------
static int failures = 0;

static void check(bool ok, const char* what, const std::string& detail = "") {
    std::printf("  %-46s %s  %s\n", what, ok ? "PASS" : "FAIL", detail.c_str());
    if (!ok) failures++;
}

static std::string fmt(const char* label, double v, const char* unit = "") {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s = %.3e %s", label, v, unit);
    return buf;
}

// -----------------------------------------------------------------------------
// Scenes whose answers are arithmetic
// -----------------------------------------------------------------------------

/// A plane at range d, tilted about y, spanning x in [x_lo, x_hi].
///
/// The tilt is applied so that a ray fired along +z from the origin strikes at exactly
/// range d with cos(theta) = cos(tilt): the plane contains (cos a, 0, sin a) and
/// (0, 1, 0), so its normal is (-sin a, 0, cos a).
static Scene make_plane(float d, float tilt_deg, float x_lo, float x_hi,
                        float y_half = 40.f, float reflectance = 0.4f) {
    Scene s;
    s.add_material(reflectance);
    const float a = tilt_deg * 3.14159265358979f / 180.f;
    const float ca = std::cos(a), sa = std::sin(a);
    auto pt = [&](float x, float y) { return Vec3{x * ca, y, d + x * sa}; };
    s.add_quad(pt(x_lo, -y_half), pt(x_hi, -y_half), pt(x_hi, y_half), pt(x_lo, y_half), 0);
    s.build();
    return s;
}

/// Two planes meeting at x = 0 with a range discontinuity: the near one occupies x > 0,
/// the far one x < 0. A beam aimed straight down +z has its footprint split exactly in
/// half by the step, which is the multi-return geometry a single-return instrument
/// cannot represent.
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

// -----------------------------------------------------------------------------
// Running one batch of beams
// -----------------------------------------------------------------------------
struct Result {
    std::vector<float>       waveforms;   // n_beams * bins
    std::vector<TruthReturn> truth;       // n_beams * max_returns
    std::vector<int>         n_returns;
};

static Result run_beams(const DeviceScene& ds, const std::vector<Beam>& beams,
                        const SensorConfig& cfg)
{
    const int n = static_cast<int>(beams.size());
    Beam* d_beams = nullptr; float* d_wave = nullptr;
    TruthReturn* d_truth = nullptr; int* d_nret = nullptr;

    CUDA_CHECK(cudaMalloc(&d_beams, n * sizeof(Beam)));
    CUDA_CHECK(cudaMalloc(&d_wave, static_cast<size_t>(n) * cfg.bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_truth, static_cast<size_t>(n) * cfg.max_returns * sizeof(TruthReturn)));
    CUDA_CHECK(cudaMalloc(&d_nret, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_beams, beams.data(), n * sizeof(Beam), cudaMemcpyHostToDevice));

    lidar_trace<<<n, 256, lidar_smem_bytes(cfg)>>>(
        ds.nodes, ds.indices, ds.tris, ds.mats, d_beams, cfg,
        d_wave, d_truth, d_nret, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Result r;
    r.waveforms.resize(static_cast<size_t>(n) * cfg.bins);
    r.truth.resize(static_cast<size_t>(n) * cfg.max_returns);
    r.n_returns.resize(n);
    CUDA_CHECK(cudaMemcpy(r.waveforms.data(), d_wave, r.waveforms.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.truth.data(), d_truth, r.truth.size() * sizeof(TruthReturn),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.n_returns.data(), d_nret, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_beams); cudaFree(d_wave); cudaFree(d_truth); cudaFree(d_nret);
    return r;
}

static Beam nadir_beam(float divergence) {
    Beam b;
    b.origin = make_float3(0.f, 0.f, 0.f);
    b.axis = make_float3(0.f, 0.f, 1.f);
    b.divergence_half_angle = divergence;
    return b;
}

/// Total energy in a waveform, and the time of its peak bin.
static double wave_energy(const float* w, int bins) {
    double e = 0.0;
    for (int k = 0; k < bins; ++k) e += w[k];
    return e;
}

static int peak_bin(const float* w, int bins) {
    int best = 0;
    for (int k = 1; k < bins; ++k) if (w[k] > w[best]) best = k;
    return best;
}

// -----------------------------------------------------------------------------

int main() {
    std::printf("\nQuBLAR Phase 1 -- LiDAR sensor model\n\n");

    SensorConfig cfg;              // 2048 bins x 0.2 ns -> 61.4 m of unambiguous range
    cfg.bins = 2048;

    const double c = kSpeedOfLight;
    const double rho = 0.4;
    const float  tiny_div = 1e-7f;   // effectively a pencil beam

    // -- analytic plane, normal incidence --------------------------------------
    {
        const float d = 10.f;
        Scene sc = make_plane(d, 0.f, -40.f, 40.f);
        DeviceScene ds; ds.upload(sc);
        Result r = run_beams(ds, {nadir_beam(tiny_div)}, cfg);

        const double e = wave_energy(r.waveforms.data(), cfg.bins);
        const double e_exact = rho / (double(d) * d);
        const double rel = std::fabs(e - e_exact) / e_exact;

        const int kp = peak_bin(r.waveforms.data(), cfg.bins);
        const double t_peak = cfg.t0_seconds + (kp + 0.5) * cfg.bin_seconds;
        const double r_peak = 0.5 * t_peak * c;

        check(rel < 2e-3, "returned energy matches rho/d^2",
              fmt("rel", rel));
        check(std::fabs(r_peak - d) < 0.5 * 0.5 * cfg.bin_seconds * c + 1e-6,
              "peak sits at the analytic range", fmt("peak", r_peak, "m"));
        ds.free();
    }

    // -- zero-divergence limit: the waveform IS the transmitted pulse ----------
    // Total energy can be right while the shape is wrong, so the shape is checked too.
    {
        const float d = 10.f;
        Scene sc = make_plane(d, 0.f, -40.f, 40.f);
        DeviceScene ds; ds.upload(sc);
        Result r = run_beams(ds, {nadir_beam(tiny_div)}, cfg);

        const double sigma = pulse_sigma(cfg);
        const double t_r = 2.0 * d / c;
        const double amp = rho / (double(d) * d);

        double max_abs = 0.0, peak = 0.0;
        for (int k = 0; k < cfg.bins; ++k) {
            const double e0 = cfg.t0_seconds + k * double(cfg.bin_seconds);
            const double e1 = e0 + cfg.bin_seconds;
            const double frac = 0.5 * (std::erf((e1 - t_r) / (sigma * std::sqrt(2.0)))
                                     - std::erf((e0 - t_r) / (sigma * std::sqrt(2.0))));
            const double want = amp * frac;
            peak = std::max(peak, want);
            max_abs = std::max(max_abs, std::fabs(double(r.waveforms[k]) - want));
        }
        check(max_abs / peak < 1e-3, "waveform equals the transmitted pulse",
              fmt("max dev / peak", max_abs / peak));
        ds.free();
    }

    // -- sub-bin phase: energy must not depend on where the return lands -------
    // A point-sampled IRF passes every check above and fails this one by percent.
    {
        const double bin_m = 0.5 * cfg.bin_seconds * c;
        double lo = 1e30, hi = -1e30;
        for (int i = 0; i < 16; ++i) {
            const float d = 10.f + float(i * bin_m / 16.0);
            Scene sc = make_plane(d, 0.f, -40.f, 40.f);
            DeviceScene ds; ds.upload(sc);
            Result r = run_beams(ds, {nadir_beam(tiny_div)}, cfg);
            const double e = wave_energy(r.waveforms.data(), cfg.bins) * double(d) * d;
            lo = std::min(lo, e); hi = std::max(hi, e);
            ds.free();
        }
        const double spread = (hi - lo) / (0.5 * (hi + lo));
        check(spread < 1e-4, "energy is independent of sub-bin phase",
              fmt("spread", spread));
    }

    // -- 1/d^2 over a range sweep ---------------------------------------------
    {
        const float ds_m[] = {5.f, 10.f, 20.f, 40.f};
        double lo = 1e30, hi = -1e30;
        std::string detail;
        for (float d : ds_m) {
            Scene sc = make_plane(d, 0.f, -40.f, 40.f);
            DeviceScene dsc; dsc.upload(sc);
            Result r = run_beams(dsc, {nadir_beam(1e-3f)}, cfg);
            const double e = wave_energy(r.waveforms.data(), cfg.bins) * double(d) * d;
            lo = std::min(lo, e); hi = std::max(hi, e);
            dsc.free();
        }
        const double spread = (hi - lo) / (0.5 * (hi + lo));
        check(spread < 5e-3, "energy falls as 1/d^2 over 5-40 m",
              fmt("spread of E*d^2", spread));
    }

    // -- cos(theta) over an incidence sweep -----------------------------------
    {
        const float tilts[] = {0.f, 15.f, 30.f, 45.f, 60.f};
        double lo = 1e30, hi = -1e30;
        for (float tilt : tilts) {
            Scene sc = make_plane(10.f, tilt, -40.f, 40.f);
            DeviceScene dsc; dsc.upload(sc);
            Result r = run_beams(dsc, {nadir_beam(1e-3f)}, cfg);
            const double ct = std::cos(tilt * 3.14159265358979 / 180.0);
            const double e = wave_energy(r.waveforms.data(), cfg.bins) / ct;
            lo = std::min(lo, e); hi = std::max(hi, e);
            dsc.free();
        }
        const double spread = (hi - lo) / (0.5 * (hi + lo));
        check(spread < 5e-3, "energy falls as cos(theta) to 60 deg",
              fmt("spread of E/cos", spread));
    }

    // -- half-plane: the beam on an edge returns exactly half its energy -------
    // Analytic by symmetry, and therefore a real convergence target. A cone sampler
    // biased toward the axis, or an orthonormal basis that flips, breaks this.
    {
        Scene sc = make_plane(10.f, 0.f, 0.f, 40.f);
        DeviceScene ds; ds.upload(sc);

        double err_coarse = 0.0, err_fine = 0.0;
        std::string trail;
        for (int S : {4, 8, 16, 32, 64}) {
            SensorConfig c2 = cfg;
            c2.samples_sqrt = S;
            Result r = run_beams(ds, {nadir_beam(2e-3f)}, c2);
            const double w = (r.n_returns[0] > 0) ? r.truth[0].weight : 0.0;
            const double err = std::fabs(w - 0.5);
            if (S == 4)  err_coarse = err;
            if (S == 64) err_fine = err;
            char b[64]; std::snprintf(b, sizeof(b), "%d:%.4f ", S, w);
            trail += b;
        }
        check(err_fine < 5e-4, "half-lit footprint returns half the energy",
              "|w-0.5| = " + std::to_string(err_fine));

        // The convergence check compares the ends of the sweep and deliberately does NOT
        // demand a monotone sequence. Stratified Monte Carlo error is a random variable:
        // it falls IN EXPECTATION, and any particular run steps up and down on the way.
        // Requiring monotonicity would produce a test that fails on some seeds and passes
        // on others, which is worse than no test -- it teaches you to ignore a red line.
        // What is checkable is that refining the estimate actually refines it.
        check(err_coarse > 4.0 * err_fine, "refining the estimate reduces the error",
              trail + "  (4 -> 64: " + std::to_string(err_coarse / err_fine) + "x)");
        ds.free();
    }

    // -- step edge: two surfaces, two returns ---------------------------------
    {
        const float d1 = 10.f, d2 = 12.f;
        Scene sc = make_step(d1, d2);
        DeviceScene ds; ds.upload(sc);
        SensorConfig c2 = cfg;
        c2.samples_sqrt = 64;
        Result r = run_beams(ds, {nadir_beam(2e-3f)}, c2);

        check(r.n_returns[0] == 2, "a step edge produces two returns",
              std::to_string(r.n_returns[0]) + " returns");

        if (r.n_returns[0] == 2) {
            const TruthReturn& a = r.truth[0];
            const TruthReturn& b = r.truth[1];
            const double dr = std::max(std::fabs(a.range - d1), std::fabs(b.range - d2));
            const double dw = std::max(std::fabs(a.weight - 0.5), std::fabs(b.weight - 0.5));
            check(dr < 1e-3, "both returns sit at their analytic ranges",
                  fmt("max |delta|", dr, "m"));
            check(dw < 5e-3, "the footprint splits evenly across the edge",
                  fmt("max |w-0.5|", dw));

            // The near surface is closer, so per unit weight it returns more energy.
            // 1/d^2 across one beam, which no single-return instrument can report.
            const double ratio = (double(d2) * d2) / (double(d1) * d1);
            std::printf("  [near/far return ratio expected %.3f from 1/d^2 alone]\n", ratio);
        }
        ds.free();
    }

    std::printf("\n%s\n\n", failures == 0 ? "all checks passed" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
