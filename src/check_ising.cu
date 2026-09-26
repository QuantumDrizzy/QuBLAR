// =============================================================================
// QuBLAR -- binary branches against their oracle, the void, and the empty
// pyramid (ADR-007)
// =============================================================================
//   A. the oracle: a 16-variable problem enumerated exhaustively (2^16
//      configurations); the annealer, cooled to MAP, must reach its ground
//      state energy before any pyramid number is reported.
//   B. the Big Void, three views, same data as check_muon: the tri-state map
//      (exists / does not exist / undecided), against truth.
//   C. hallucination: the empty pyramid, same seeds, same lambda and kappa,
//      must yield NO "does not exist" voxel.
//   D. data or prior: the same void run with lambda = kappa = 0; voids that
//      exist only with the prior on are counted as prior-driven.
//   E. calibration: among undecided voxels, is the void fraction where p says?
// Exports the maps to build\ising_out\ for tooling (ndim render, Blaze).
// =============================================================================

#include "muon.cuh"
#include "muon_recon.hpp"
#include "muon_replica.hpp"
#include "ising_recon.hpp"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <cstdlib>
#include <cstdio>
#include <fstream>
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
    std::snprintf(buf, sizeof(buf), "%s = %.4g", label, v);
    return buf;
}

// The declared priors of this experiment (ADR-007 §1):
static const double kPriorVoidFraction = 1e-3;   // a voxel is void 1 time in 1000
static const double kLambda = 2.0;               // Ising coupling of the wall prior
static const int kBranches = 16;

static double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Default exposure 2^27 muons per chamber (4x check_muon's). The measured
// sweep (RESULTS-phase6): at 2^25 the data give 303 nats for the true void
// against the prior's 421, so the posterior correctly declines it; from 2^26
// the data win and the branches localise it. Pass log2(N) to rerun any point.
static int g_candidates = 1 << 27;

int main(int argc, char** argv) {
    // optional: exposure as log2(candidates per chamber), for the exposure sweep
    if (argc > 1) g_candidates = 1 << std::atoi(argv[1]);
    std::printf("\nQuBLAR -- binary branches (ADR-007)\n\n");
    const unsigned n_threads =
        std::max(1u, std::min(16u, std::thread::hardware_concurrency() > 2
                                       ? std::thread::hardware_concurrency() - 2 : 1u));

    // ---- A. the oracle -------------------------------------------------------
    {
        std::printf("  A. the oracle\n");
        BinaryProblem p;
        const int n = 16, n_rays = 12;
        uint64_t s = 7;
        p.lambda = 0.7;
        p.kappa = 0.4;
        for (int i = 0; i < n; ++i) { p.var_voxel.push_back(i); }
        p.voxel_var = p.var_voxel;
        for (int b = 0; b < n_rays; ++b) {
            p.d.push_back(detail::uniform(s) * 0.3);
            p.w.push_back(20.0 + 80.0 * detail::uniform(s));
        }
        p.row_ptr.push_back(0);
        for (int i = 0; i < n; ++i) {
            for (int b = 0; b < n_rays; ++b)
                if (detail::uniform(s) < 0.35) {
                    p.ray.push_back(b);
                    p.a.push_back(float(0.02 + 0.1 * detail::uniform(s)));
                }
            p.row_ptr.push_back(int(p.ray.size()));
        }
        p.nbr_ptr.push_back(0);
        for (int i = 0; i < n; ++i) {            // a 4x4 grid
            const int x = i % 4, y = i / 4;
            if (x > 0) p.nbr.push_back(i - 1);
            if (x < 3) p.nbr.push_back(i + 1);
            if (y > 0) p.nbr.push_back(i - 4);
            if (y < 3) p.nbr.push_back(i + 4);
            p.nbr_ptr.push_back(int(p.nbr.size()));
        }
        double exact = 1e300;
        std::vector<uint8_t> x(n);
        for (uint32_t c = 0; c < (1u << n); ++c) {
            for (int i = 0; i < n; ++i) x[i] = (c >> i) & 1u;
            exact = std::min(exact, binary_energy(p, x));
        }
        Schedule map;
        map.t_hot = 5.0;
        map.t_cold = 1e-3;
        map.anneal_sweeps = 400;
        map.hold_sweeps = 20;
        double annealed = 1e300;
        for (int seed = 1; seed <= 8; ++seed)
            annealed = std::min(annealed, binary_energy(p, anneal_branch(p, map, seed)));
        check(std::fabs(annealed - exact) < 1e-9, "annealer reaches the exact ground state",
              num("annealed", annealed) + ", " + num("exact (2^16)", exact));
    }

    // ---- the replica: same pyramid, chambers and exposure as check_muon ----------
    std::vector<float> host_mu, host_mu_empty;
    float *d_mu = nullptr, *d_mu_empty = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mu, size_t(kN) * kN * kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mu_empty, size_t(kN) * kN * kN * sizeof(float)));
    const VoxelMedium m_void = make_medium(true, host_mu, d_mu);
    const VoxelMedium m_empty = make_medium(false, host_mu_empty, d_mu_empty);
    VoxelMedium m_model = m_empty;
    m_model.mu = host_mu_empty.data();          // the surveyed shape, solid rock

    // Unknowns: interior voxels, minus the detector rooms. A voxel holding a
    // chamber is crossed by every ray of that chamber's sky, so it acts as a
    // global offset and absorbs any normalisation noise as a "void" (measured:
    // one false void 3 m from chamber 2). Where the detectors sit is surveyed,
    // not inferred. [Changed after that false void was seen.]
    std::vector<char> domain(host_mu_empty.size(), 0);
    for (int k = 0; k < kN; ++k)
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kN; ++i) {
                const size_t v = (size_t(k) * kN + j) * kN + i;
                if (host_mu_empty[v] < kMuRock) continue;
                const float x = kLo.x + (i + 0.5f) * kVoxel;
                const float y = kLo.y + (j + 0.5f) * kVoxel;
                const float z = kLo.z + (k + 0.5f) * kVoxel;
                bool room = false;
                for (const float3& c : kChambers) {
                    const float dx = x - c.x, dy = y - c.y, dz = z - c.z;
                    room |= dx * dx + dy * dy + dz * dz < 3.0f * 3.0f;
                }
                domain[v] = room ? 0 : 1;
            }
    std::vector<char> truth(host_mu.size(), 0);
    for (size_t v = 0; v < truth.size(); ++v) truth[v] = domain[v] && host_mu[v] < kMuRock;

    std::vector<MuonBinnedData> seen, seen0;
    auto te = std::chrono::steady_clock::now();
    for (int c = 0; c < 3; ++c) {
        seen.push_back(expose(m_void, kChambers[c], 1u + c, imaging_sky(), g_candidates));
        seen0.push_back(expose(m_empty, kChambers[c], 1u + c, imaging_sky(), g_candidates));
    }
    std::printf("\n  exposure: %d candidates per chamber (2^%d), 6 exposures in %.1f s\n",
                g_candidates, int(std::log2(double(g_candidates)) + 0.5), seconds_since(te));
    const std::vector<MuonView> views = {{kChambers[0], &seen[0]}, {kChambers[1], &seen[1]},
                                         {kChambers[2], &seen[2]}};
    const std::vector<MuonView> views0 = {{kChambers[0], &seen0[0]}, {kChambers[1], &seen0[1]},
                                          {kChambers[2], &seen0[2]}};
    const double kappa = std::log((1.0 - kPriorVoidFraction) / kPriorVoidFraction);
    Schedule posterior;   // anneal to T = 1 and hold: posterior samples

    // ---- B. the Big Void -----------------------------------------------------
    std::printf("\n  B. the Big Void (lambda = %.2f, kappa = ln((1-p0)/p0) = %.2f, p0 = %g, "
                "%d branches)\n", kLambda, kappa, kPriorVoidFraction, kBranches);
    auto t0 = std::chrono::steady_clock::now();
    const BinaryProblem prob = build_binary_problem(m_model, views, domain, kLambda, kappa);
    std::printf("    %d variables, %d rays, %zu ray-voxel entries (built in %.1f s)\n",
                prob.n_vars(), prob.n_rays(), prob.a.size(), seconds_since(t0));
    t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<uint8_t>> branches;
    const std::vector<float> pv = branch_fractions(prob, posterior, kBranches, n_threads,
                                                   &branches);
    std::printf("    %d branches annealed in %.1f s on %u threads\n", kBranches,
                seconds_since(t0), n_threads);

    int n_not = 0, n_und = 0, hit = 0, truth_n = 0;
    double cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < prob.n_vars(); ++i) {
        const int v = prob.var_voxel[i];
        truth_n += truth[v];
        const Bit bit = classify(pv[i]);
        if (bit == Bit::NotThere) {
            ++n_not;
            hit += truth[v];
            cx += kLo.x + (v % kN + 0.5f) * kVoxel;
            cy += kLo.y + ((v / kN) % kN + 0.5f) * kVoxel;
            cz += kLo.z + (v / (kN * kN) + 0.5f) * kVoxel;
        } else if (bit == Bit::Undecided) {
            ++n_und;
        }
    }
    const double iou = double(hit) / double(n_not + truth_n - hit);
    double dist = -1.0;
    if (n_not > 0) {
        cx /= n_not; cy /= n_not; cz /= n_not;
        const double dy = std::max(0.0, std::fabs(cy) - kVoidHY);
        dist = std::sqrt(cx * cx + dy * dy + (cz - kVoidZ) * (cz - kVoidZ));
    }
    std::printf("    truth: %d void voxels; found 'does not exist': %d (%d correct), "
                "undecided: %d\n", truth_n, n_not, hit, n_und);
    if (n_not > 0)
        std::printf("    centroid of the found void (%+.1f, %+.1f, %+.1f) m, %.1f m from "
                    "the void's axis\n", cx, cy, cz, dist);
    std::printf("    baseline (ADR-006 MLEM, same data): deficit peak 30.0 m off\n");
    check(n_not > 0 && dist >= 0.0 && dist < 6.0, "binary branches localise the void",
          num("distance m", dist) + ", " + num("IoU", iou));

    // ---- C. hallucination ------------------------------------------------------
    std::printf("\n  C. the empty pyramid, same seeds, same priors\n");
    const BinaryProblem prob0 = build_binary_problem(m_model, views0, domain, kLambda, kappa);
    const std::vector<float> pv0 = branch_fractions(prob0, posterior, kBranches, n_threads);
    int false_not = 0, und0 = 0;
    for (int i = 0; i < prob0.n_vars(); ++i) {
        const Bit bit = classify(pv0[i]);
        false_not += bit == Bit::NotThere;
        und0 += bit == Bit::Undecided;
    }
    std::printf("    'does not exist' voxels: %d, undecided: %d\n", false_not, und0);
    check(false_not == 0, "no void is found where there is none",
          num("false voids", false_not));

    // ---- D. data or prior ------------------------------------------------------
    std::printf("\n  D. the void run with the prior off (lambda = kappa = 0)\n");
    const BinaryProblem prob_d = build_binary_problem(m_model, views, domain, 0.0, 0.0);
    {
        // The evidence budget: what the data pay for the TRUE void (data energy
        // of the truth minus that of solid rock, in nats) against what the
        // declared prior charges for it (kappa per voxel + lambda per face).
        std::vector<uint8_t> xt(prob_d.n_vars(), 0), x0(prob_d.n_vars(), 0);
        for (int i = 0; i < prob_d.n_vars(); ++i) xt[i] = truth[prob_d.var_voxel[i]];
        const double gain = binary_energy(prob_d, x0) - binary_energy(prob_d, xt);
        BinaryProblem prior_only = prob;           // same structure, data switched off
        std::fill(prior_only.w.begin(), prior_only.w.end(), 0.0);
        const double cost = binary_energy(prior_only, xt) - binary_energy(prior_only, x0);
        std::printf("    evidence budget for the true void: data %+.1f nats, prior %+.1f nats "
                    "-> the prior %s\n", gain, cost, gain > cost ? "is outweighed" : "wins");
    }
    const std::vector<float> pv_d = branch_fractions(prob_d, posterior, kBranches, n_threads);
    int data_driven = 0, prior_hides = 0;
    double pv_truth = 0.0, pvd_truth = 0.0;
    for (int i = 0; i < prob.n_vars(); ++i) {
        if (classify(pv[i]) == Bit::NotThere && pv_d[i] > 0.5f) ++data_driven;
        if (classify(pv[i]) == Bit::Exists && pv_d[i] > 0.5f) ++prior_hides;
        if (truth[prob.var_voxel[i]]) { pv_truth += pv[i]; pvd_truth += pv_d[i]; }
    }
    std::printf("    of %d 'does not exist' voxels, %d stay above p = 0.5 without the prior "
                "(data-driven); %d need it (prior-driven)\n",
                n_not, data_driven, n_not - data_driven);
    std::printf("    'exists' only because of the prior (p > 0.5 without it): %d voxels\n",
                prior_hides);
    std::printf("    mean p_void over the true void: %.3f with the prior, %.3f without\n",
                pv_truth / std::max(1, truth_n), pvd_truth / std::max(1, truth_n));

    // ---- E. calibration --------------------------------------------------------
    std::printf("\n  E. calibration against truth\n");
    const float edges[5] = {0.0f, 0.1f, 0.5f, 0.9f, 1.01f};
    for (int k = 0; k < 4; ++k) {
        int n = 0, t = 0;
        for (int i = 0; i < prob.n_vars(); ++i)
            if (pv[i] >= edges[k] && pv[i] < edges[k + 1]) {
                ++n;
                t += truth[prob.var_voxel[i]];
            }
        std::printf("    p in [%.1f, %.1f): %7d voxels, truly void %.3f\n", edges[k],
                    std::min(1.0f, edges[k + 1]), n, n ? double(t) / n : 0.0);
    }

    // ---- F. efficiency: the same answer for less work --------------------------
    // Owner's rule: a result that holds at 50 % of the compute should not be
    // bought at 100 %. Shorter schedules and fewer branches are measured
    // against the default run above; a configuration counts as "same answer"
    // only if its 'does not exist' set is IDENTICAL, voxel for voxel.
    std::printf("\n  F. efficiency (reference: %d branches, %d + %d sweeps)\n",
                kBranches, posterior.anneal_sweeps, posterior.hold_sweeps);
    {
        std::vector<int> ref;
        for (int i = 0; i < prob.n_vars(); ++i)
            if (classify(pv[i]) == Bit::NotThere) ref.push_back(i);
        struct Cfg { int branches, anneal, hold; };
        const Cfg cfgs[] = {{16, 100, 25}, {16, 50, 15}, {8, 200, 50}, {8, 100, 25},
                            {8, 50, 15}, {4, 100, 25}};
        std::printf("    %-9s %-15s %8s %10s %9s %s\n", "branches", "sweeps", "time s",
                    "work (%)", "found", "same set as reference");
        const double ref_work = double(kBranches) * (posterior.anneal_sweeps + posterior.hold_sweeps);
        for (const Cfg& cf : cfgs) {
            Schedule s = posterior;
            s.anneal_sweeps = cf.anneal;
            s.hold_sweeps = cf.hold;
            const auto tc = std::chrono::steady_clock::now();
            const std::vector<float> q = branch_fractions(prob, s, cf.branches, n_threads);
            const double secs = seconds_since(tc);
            std::vector<int> got;
            int correct = 0;
            for (int i = 0; i < prob.n_vars(); ++i)
                if (classify(q[i]) == Bit::NotThere) {
                    got.push_back(i);
                    correct += truth[prob.var_voxel[i]];
                }
            const double work = 100.0 * cf.branches * (cf.anneal + cf.hold) / ref_work;
            char sweeps[32];
            std::snprintf(sweeps, sizeof(sweeps), "%d + %d", cf.anneal, cf.hold);
            std::printf("    %-9d %-15s %8.1f %9.1f%% %4d (%d ok) %s\n", cf.branches, sweeps,
                        secs, work, int(got.size()), correct, got == ref ? "yes" : "no");
        }
    }

    // ---- G. the freeze: provable rock is not annealed ---------------------------
    {
        const double kMargin = 10.0;   // nats: a frozen flip has p <= e^-10 per sweep
        std::printf("\n  G. freezing provable rock (margin %.0f nats)\n", kMargin);
        std::vector<int> ref;
        for (int i = 0; i < prob.n_vars(); ++i)
            if (classify(pv[i]) == Bit::NotThere) ref.push_back(i);
        // the yardstick: the reference itself with other seeds
        const std::vector<float> pv_other = branch_fractions(prob, posterior, kBranches,
                                                             n_threads, nullptr, 1000);
        std::vector<int> other;
        for (int i = 0; i < prob.n_vars(); ++i)
            if (classify(pv_other[i]) == Bit::NotThere) other.push_back(i);
        const auto jaccard = [](const std::vector<int>& a, const std::vector<int>& b) {
            std::vector<int> both;
            std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(both));
            const double uni = double(a.size() + b.size() - both.size());
            return uni > 0 ? both.size() / uni : 1.0;
        };
        auto tf = std::chrono::steady_clock::now();
        const FreezeResult fr = freeze_provable_rock(prob, kMargin);
        const double t_freeze = seconds_since(tf);
        tf = std::chrono::steady_clock::now();
        const std::vector<float> pv_r = branch_fractions(fr.reduced, posterior, kBranches, n_threads);
        const double t_anneal = seconds_since(tf);
        std::vector<int> got;
        int correct = 0;
        for (int j = 0; j < fr.reduced.n_vars(); ++j)
            if (classify(pv_r[j]) == Bit::NotThere) {
                got.push_back(fr.kept[j]);
                correct += truth[fr.reduced.var_voxel[j]];
            }
        std::sort(got.begin(), got.end());
        std::printf("    variables annealed: %d of %d (%.2f%%), %d freeze rounds, %.2f s to prove\n",
                    fr.reduced.n_vars(), prob.n_vars(), 100.0 * fr.reduced.n_vars() / prob.n_vars(),
                    fr.rounds, t_freeze);
        std::printf("    %d branches: %.2f s (reference %.1f s); found %d (%d correct)\n",
                    kBranches, t_anneal, 6.9, int(got.size()), correct);
        std::printf("    Jaccard vs reference: frozen run %.3f | same engine, other seeds %.3f\n",
                    jaccard(got, ref), jaccard(other, ref));
        check(jaccard(got, ref) >= jaccard(other, ref) - 0.15 && correct == int(got.size()),
              "freezing keeps the answer within seed-to-seed spread",
              num("frozen", jaccard(got, ref)) + ", " + num("seeds", jaccard(other, ref)));
    }

    // ---- export ----------------------------------------------------------------
    {
        std::vector<float> grid(size_t(kN) * kN * kN, -1.0f);   // -1: outside the domain
        std::vector<float> grid_d(grid.size(), -1.0f);
        for (int i = 0; i < prob.n_vars(); ++i) {
            grid[prob.var_voxel[i]] = pv[i];
            grid_d[prob.var_voxel[i]] = pv_d[i];
        }
        std::ofstream f("build\\ising_out_pvoid.bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(grid.data()), grid.size() * sizeof(float));
        std::ofstream fd("build\\ising_out_pvoid_noprior.bin", std::ios::binary);
        fd.write(reinterpret_cast<const char*>(grid_d.data()), grid_d.size() * sizeof(float));
        std::ofstream ft("build\\ising_out_truth.bin", std::ios::binary);
        ft.write(truth.data(), truth.size());
        std::ofstream fb("build\\ising_out_branches.bin", std::ios::binary);
        for (const auto& x : branches) fb.write(reinterpret_cast<const char*>(x.data()), x.size());
        std::ofstream meta("build\\ising_out.meta");
        meta << "nx " << kN << "\nvoxel " << kVoxel << "\nlo " << kLo.x << " " << kLo.y << " "
             << kLo.z << "\nbranches " << kBranches << "\nvars " << prob.n_vars()
             << "\nlambda " << kLambda << "\nkappa " << kappa << "\n";
        // the region of interest for tensor tooling (Blaze): the 20 variables
        // with the highest p_void, conditioned on the branch consensus elsewhere
        {
            std::vector<int> order(prob.n_vars());
            for (int i = 0; i < prob.n_vars(); ++i) order[i] = i;
            std::partial_sort(order.begin(), order.begin() + 20, order.end(),
                              [&](int u, int v) { return pv[u] > pv[v]; });
            std::vector<int> roi(order.begin(), order.begin() + 20);
            std::sort(roi.begin(), roi.end());
            std::vector<uint8_t> x_bar(prob.n_vars());
            for (int i = 0; i < prob.n_vars(); ++i) x_bar[i] = pv[i] >= 0.5f;
            const RoiQubo q = roi_qubo(prob, roi, x_bar);
            // oracle: E(x) - E(0) from the full energy equals the QUBO, for a
            // few random x_R -- the expansion is checked, not trusted
            double worst = 0.0;
            std::vector<uint8_t> xf = x_bar;
            for (int v : roi) xf[v] = 0;
            const double e0 = binary_energy(prob, xf);
            uint64_t st = 99;
            for (int trial = 0; trial < 8; ++trial) {
                std::vector<uint8_t> xr(roi.size());
                for (auto& b : xr) b = detail::uniform(st) < 0.5;
                for (size_t i = 0; i < roi.size(); ++i) xf[roi[i]] = xr[i];
                double eq = 0.0;
                for (size_t i = 0; i < roi.size(); ++i) {
                    eq += q.h[i] * xr[i];
                    for (size_t j = i + 1; j < roi.size(); ++j)
                        eq += q.J[i * roi.size() + j] * xr[i] * xr[j];
                }
                worst = std::max(worst, std::fabs((binary_energy(prob, xf) - e0) - eq));
            }
            check(worst < 1e-6 * std::max(1.0, std::fabs(e0)),
                  "ROI QUBO equals the full energy (8 random x)", num("worst abs err", worst));
            // a second region: the 12 variables whose branches disagree most
            // (p closest to 1/2) -- where the branches really compete
            {
                std::vector<int> ord(prob.n_vars());
                for (int i = 0; i < prob.n_vars(); ++i) ord[i] = i;
                std::partial_sort(ord.begin(), ord.begin() + 12, ord.end(), [&](int u, int v) {
                    return std::fabs(pv[u] - 0.5f) < std::fabs(pv[v] - 0.5f);
                });
                std::vector<int> ru(ord.begin(), ord.begin() + 12);
                std::sort(ru.begin(), ru.end());
                const RoiQubo qu = roi_qubo(prob, ru, x_bar);
                std::ofstream fu("build\\ising_out_roi_uncertain.txt");
                fu.precision(17);
                fu << ru.size() << "\n";
                for (size_t i = 0; i < ru.size(); ++i)
                    fu << prob.var_voxel[ru[i]] << " " << pv[ru[i]] << " "
                       << int(truth[prob.var_voxel[ru[i]]]) << " " << qu.h[i] << "\n";
                for (size_t i = 0; i < ru.size(); ++i) {
                    for (size_t j = 0; j < ru.size(); ++j) fu << qu.J[i * ru.size() + j] << " ";
                    fu << "\n";
                }
            }
            std::ofstream fr("build\\ising_out_roi.txt");
            fr.precision(17);
            fr << roi.size() << "\n";
            for (size_t i = 0; i < roi.size(); ++i)
                fr << prob.var_voxel[roi[i]] << " " << pv[roi[i]] << " "
                   << int(truth[prob.var_voxel[roi[i]]]) << " " << q.h[i] << "\n";
            for (size_t i = 0; i < roi.size(); ++i) {
                for (size_t j = 0; j < roi.size(); ++j) fr << q.J[i * roi.size() + j] << " ";
                fr << "\n";
            }
        }
        std::printf("\n    exported build\\ising_out_* (p_void grids, truth, branches, meta)\n");
    }

    cudaFree(d_mu);
    cudaFree(d_mu_empty);
    std::printf("\n%s\n\n", failures == 0 ? "all checks passed" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
