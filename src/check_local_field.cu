// =============================================================================
// QuBLAR -- host check: local-field ΔE matches binary_energy difference
// =============================================================================
// The annealing sweep's local field scores a proposed flip without flipping.
// This check compares local_field_dE to binary_energy(after) - binary_energy(before)
// on a tiny synthetic BinaryProblem. No pyramid anneal. Schedule defaults untouched.
// Metropolis accept/reject is not under test here and is not a LYTH kernel.
// =============================================================================

#include "ising_recon.hpp"
#include "run_talk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace argos;

static int failures = 0;

static void check(bool ok, const char* what, const std::string& detail = "") {
    std::printf("  %-50s %s  %s\n", what, ok ? "PASS" : "FAIL", detail.c_str());
    if (!ok) failures++;
}

static std::string num(const char* label, double v) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s = %.6g", label, v);
    return buf;
}

static std::vector<double> residuals_for(const BinaryProblem& p,
                                         const std::vector<uint8_t>& x) {
    std::vector<double> r(p.n_rays());
    for (int b = 0; b < p.n_rays(); ++b) r[b] = -p.d[b];
    for (int i = 0; i < p.n_vars(); ++i)
        if (x[i])
            for (int k = p.row_ptr[i]; k < p.row_ptr[i + 1]; ++k) r[p.ray[k]] += p.a[k];
    return r;
}

static BinaryProblem tiny_synthetic() {
    BinaryProblem p;
    const int n = 6, n_rays = 5;
    uint64_t s = 11;
    p.lambda = 0.7;
    p.kappa = 0.4;
    for (int i = 0; i < n; ++i) p.var_voxel.push_back(i);
    p.voxel_var = p.var_voxel;
    for (int b = 0; b < n_rays; ++b) {
        p.d.push_back(detail::uniform(s) * 0.4);
        p.w.push_back(10.0 + 40.0 * detail::uniform(s));
    }
    p.row_ptr.push_back(0);
    for (int i = 0; i < n; ++i) {
        for (int b = 0; b < n_rays; ++b)
            if (detail::uniform(s) < 0.5) {
                p.ray.push_back(b);
                p.a.push_back(float(0.03 + 0.12 * detail::uniform(s)));
            }
        p.row_ptr.push_back(int(p.ray.size()));
    }
    // line of 6: each has one or two neighbours
    p.nbr_ptr.push_back(0);
    for (int i = 0; i < n; ++i) {
        if (i > 0) p.nbr.push_back(i - 1);
        if (i < n - 1) p.nbr.push_back(i + 1);
        p.nbr_ptr.push_back(int(p.nbr.size()));
    }
    return p;
}

int main() {
    RunTalk talk = RunTalk::begin("check_local_field");
    std::printf("\nQuBLAR -- local-field ΔE vs binary_energy\n\n");
    const BinaryProblem p = tiny_synthetic();
    const int n = p.n_vars();

    // Configurations: all-rock, a mixed pattern, and all-void (covers both
    // flip directions across variables without running an anneal).
    std::vector<std::vector<uint8_t>> configs;
    configs.push_back(std::vector<uint8_t>(n, 0));
    {
        std::vector<uint8_t> mixed(n, 0);
        mixed[0] = 1;
        mixed[2] = 1;
        mixed[3] = 1;
        configs.push_back(mixed);
    }
    configs.push_back(std::vector<uint8_t>(n, 1));

    double worst = 0.0;
    double scale = 0.0;
    bool saw_uphill = false;
    int compared = 0;

    {
    Tick cpu(talk, false);
    for (const auto& x0 : configs) {
        const double e_before = binary_energy(p, x0);
        scale = std::max(scale, std::fabs(e_before));
        const std::vector<double> r = residuals_for(p, x0);
        for (int i = 0; i < n; ++i) {
            const double dE = local_field_dE(p, x0, r, i);
            if (dE > 0.0) saw_uphill = true;
            std::vector<uint8_t> x1 = x0;
            x1[i] ^= 1u;
            const double dE_exact = binary_energy(p, x1) - e_before;
            const double err = std::fabs(dE - dE_exact);
            worst = std::max(worst, err);
            ++compared;
        }
    }
    }

    const double tol = 1e-6 * std::max(1.0, scale);
    check(compared > 0, "compared proposed flips", num("n", compared));
    check(saw_uphill, "at least one uphill flip (dE > 0)",
          num("worst abs err", worst));
    check(worst < tol, "local_field_dE matches energy difference",
          num("worst abs err", worst) + ", " + num("tol", tol) + ", " +
              num("scale", scale));

    std::printf("\n  worst abs err = %.6g  (tol = %.6g)\n", worst, tol);
    std::printf("  %s\n\n", failures ? "FAILED" : "all checks passed");
    talk.end(failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
