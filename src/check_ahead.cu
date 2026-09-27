// =============================================================================
// QuBLAR -- capability: cavity a few metres ahead (synthetic)
// =============================================================================
// Policy on the existing tri-state map (classify unchanged):
//   Exists (rock, p <= 0.1)           = traversable
//   Undecided OR NotThere             = not traversable
// Declared distance: 5 m ahead. Same prior / schedule / thresholds as the void
// path in check_ising. If data do not pay: no confident void on the truth;
// undecided still blocks. Days are not claimed without I and T.
// =============================================================================

#include "muon.cuh"
#include "muon_recon.hpp"
#include "muon_replica.hpp"
#include "ising_recon.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace argos;

static const int kAheadN = 40;
static const float kAheadVoxel = 1.0f;
static const float3 kAheadLo = make_float3(-20.0f, -20.0f, -2.0f);
static const float kRockHalf = 16.0f;
static const float kRockTop = 30.0f;
static const float kAheadM = 5.0f;          // declared "few metres"
static const float kCavHalf = 1.5f;         // 3 m cube cavity
static const float3 kAheadChamber = make_float3(0.0f, 0.0f, 4.0f);
// Cavity centre: kAheadM metres ahead in +x, above the chamber so upward rays see it.
static const float3 kCavity = make_float3(kAheadM, 0.0f, 12.0f);

static const double kPriorVoidFraction = 1e-3;
static const double kLambda = 2.0;
static const int kBranches = 16;

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

static double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static void evidence_budget(const BinaryProblem& data_off, const BinaryProblem& prior,
                            const std::vector<char>& truth, double& gain, double& cost) {
    std::vector<uint8_t> xt(data_off.n_vars(), 0), x0(data_off.n_vars(), 0);
    for (int i = 0; i < data_off.n_vars(); ++i) xt[i] = truth[data_off.var_voxel[i]];
    gain = binary_energy(data_off, x0) - binary_energy(data_off, xt);
    BinaryProblem prior_only = prior;
    std::fill(prior_only.w.begin(), prior_only.w.end(), 0.0);
    cost = binary_energy(prior_only, xt) - binary_energy(prior_only, x0);
}

static bool traversable(Bit b) { return b == Bit::Exists; }

static bool in_rock(float x, float y, float z) {
    return z >= 0.0f && z <= kRockTop && fabsf(x) <= kRockHalf && fabsf(y) <= kRockHalf;
}

static bool in_cavity(float x, float y, float z) {
    return fabsf(x - kCavity.x) <= kCavHalf && fabsf(y - kCavity.y) <= kCavHalf
        && fabsf(z - kCavity.z) <= kCavHalf;
}

static VoxelMedium make_ahead(bool with_cavity, std::vector<float>& host_mu, float* d_mu) {
    host_mu.assign(size_t(kAheadN) * kAheadN * kAheadN, kMuRock * (1.2f / 2700.0f));
    for (int k = 0; k < kAheadN; ++k)
        for (int j = 0; j < kAheadN; ++j)
            for (int i = 0; i < kAheadN; ++i) {
                const float x = kAheadLo.x + (i + 0.5f) * kAheadVoxel;
                const float y = kAheadLo.y + (j + 0.5f) * kAheadVoxel;
                const float z = kAheadLo.z + (k + 0.5f) * kAheadVoxel;
                const size_t idx = (size_t(k) * kAheadN + j) * kAheadN + i;
                if (!in_rock(x, y, z)) continue;
                host_mu[idx] = kMuRock;
                if (with_cavity && in_cavity(x, y, z))
                    host_mu[idx] = kMuRock * (1.2f / 2700.0f);
                const float dx = x - kAheadChamber.x, dy = y - kAheadChamber.y, dz = z - kAheadChamber.z;
                if (dx * dx + dy * dy + dz * dz < 3.0f * 3.0f)
                    host_mu[idx] = kMuRock * (1.2f / 2700.0f);
            }
    VoxelMedium m;
    m.lo = kAheadLo;
    m.voxel = kAheadVoxel;
    m.nx = m.ny = m.nz = kAheadN;
    m.mu_rock = kMuRock;
    m.mu_air = kMuRock * (1.2f / 2700.0f);
    m.mu = d_mu;
    CUDA_CHECK(cudaMemcpy(d_mu, host_mu.data(), host_mu.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    return m;
}

static void fill_domain(const std::vector<float>& host_solid, std::vector<char>& domain) {
    domain.assign(host_solid.size(), 0);
    for (int k = 0; k < kAheadN; ++k)
        for (int j = 0; j < kAheadN; ++j)
            for (int i = 0; i < kAheadN; ++i) {
                const size_t v = (size_t(k) * kAheadN + j) * kAheadN + i;
                if (host_solid[v] < kMuRock * 0.5f) continue;
                const float x = kAheadLo.x + (i + 0.5f) * kAheadVoxel;
                const float y = kAheadLo.y + (j + 0.5f) * kAheadVoxel;
                const float z = kAheadLo.z + (k + 0.5f) * kAheadVoxel;
                const float dx = x - kAheadChamber.x, dy = y - kAheadChamber.y, dz = z - kAheadChamber.z;
                const bool room = dx * dx + dy * dy + dz * dz < 3.0f * 3.0f;
                domain[v] = room ? 0 : 1;
            }
}

int main(int argc, char** argv) {
    std::printf("\nQuBLAR -- check_ahead (cavity %.0f m ahead, traversable policy)\n\n",
                kAheadM);
    std::printf("  policy: Exists = traversable; Undecided|NotThere = not traversable\n");
    std::printf("  prior p0 = %g, lambda = %.2f; classify unchanged; rooms 3 m\n",
                kPriorVoidFraction, kLambda);
    std::printf("  days are not claimed (candidate counts only)\n\n");

    const int candidates = argc > 1 ? (1 << std::atoi(argv[1])) : (1 << 26);
    const unsigned n_threads =
        std::max(1u, std::min(16u, std::thread::hardware_concurrency() > 2
                                       ? std::thread::hardware_concurrency() - 2
                                       : 1u));
    const double kappa = std::log((1.0 - kPriorVoidFraction) / kPriorVoidFraction);
    Schedule posterior;
    const int log2e = int(std::log2(double(candidates)) + 0.5);

    std::vector<float> host_cav, host_empty;
    float *d_cav = nullptr, *d_empty = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cav, size_t(kAheadN) * kAheadN * kAheadN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_empty, size_t(kAheadN) * kAheadN * kAheadN * sizeof(float)));
    const VoxelMedium m_cav = make_ahead(true, host_cav, d_cav);
    const VoxelMedium m_empty = make_ahead(false, host_empty, d_empty);
    VoxelMedium m_model = m_empty;
    m_model.mu = host_empty.data();

    std::vector<char> domain;
    fill_domain(host_empty, domain);
    std::vector<char> truth(host_cav.size(), 0);
    for (size_t v = 0; v < truth.size(); ++v)
        truth[v] = domain[v] && host_cav[v] < kMuRock * 0.5f;

    auto t0 = std::chrono::steady_clock::now();
    MuonBinnedData seen = expose(m_cav, kAheadChamber, 1u, imaging_sky(), candidates);
    MuonBinnedData seen0 = expose(m_empty, kAheadChamber, 2u, imaging_sky(), candidates);
    std::printf("  exposure: 2^%d = %d candidates (%.1f s)\n", log2e, candidates,
                seconds_since(t0));

    const std::vector<MuonView> views = {{kAheadChamber, &seen}};
    const std::vector<MuonView> views0 = {{kAheadChamber, &seen0}};

    // ---- scene with cavity ---------------------------------------------------
    std::printf("\n  A. cavity %.0f m ahead (centre at x = %.0f m)\n", kAheadM, kCavity.x);
    const BinaryProblem prob = build_binary_problem(m_model, views, domain, kLambda, kappa);
    const BinaryProblem prob_d = build_binary_problem(m_model, views, domain, 0.0, 0.0);
    double gain = 0, cost = 0;
    evidence_budget(prob_d, prob, truth, gain, cost);
    const bool data_pay = gain > cost;
    std::printf("    %d variables, %d rays\n", prob.n_vars(), prob.n_rays());
    std::printf("    evidence: data %+.1f nats, prior %+.1f nats -> %s\n", gain, cost,
                data_pay ? "data pay" : "DECLINE (prior wins)");

    t0 = std::chrono::steady_clock::now();
    const std::vector<float> pv = branch_fractions(prob, posterior, kBranches, n_threads);
    std::printf("    %d branches in %.1f s\n", kBranches, seconds_since(t0));

    int truth_n = 0, hit = 0, n_block = 0, n_clear_on_truth = 0;
    double cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < prob.n_vars(); ++i) {
        const int v = prob.var_voxel[i];
        if (!truth[v]) continue;
        ++truth_n;
        const Bit b = classify(pv[i]);
        if (b == Bit::NotThere) {
            ++hit;
            cx += kAheadLo.x + (v % kAheadN + 0.5f) * kAheadVoxel;
            cy += kAheadLo.y + ((v / kAheadN) % kAheadN + 0.5f) * kAheadVoxel;
            cz += kAheadLo.z + (v / (kAheadN * kAheadN) + 0.5f) * kAheadVoxel;
        }
        if (!traversable(b)) ++n_block;
        else ++n_clear_on_truth;
    }
    int n_not_all = 0;
    for (int i = 0; i < prob.n_vars(); ++i)
        if (classify(pv[i]) == Bit::NotThere) ++n_not_all;

    double dist = -1.0;
    if (hit > 0) {
        cx /= hit;
        cy /= hit;
        cz /= hit;
        const double dx = cx - kCavity.x, dy = cy - kCavity.y, dz = cz - kCavity.z;
        dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    std::printf("    truth cavity voxels in domain: %d; NotThere on truth: %d; "
                "NotThere anywhere: %d\n",
                truth_n, hit, n_not_all);
    std::printf("    policy on truth cavity: %d block, %d clear (Exists)\n", n_block,
                n_clear_on_truth);
    if (data_pay && hit > 0)
        std::printf("    localisation: centroid %.2f m from cavity centre\n", dist);

    // When the data pay: every truth cavity voxel must block (Exists would green-
    // light travel into the hole), and the NotThere set must localise within 6 m.
    // When they do not: same rule as the void path — no confident void on truth;
    // prior→Exists is the correct unpaid posterior. "Undecided still blocks" is
    // the policy map, not a demand that unpaid runs produce Undecided.
    if (data_pay) {
        check(n_clear_on_truth == 0, "cavity voxels are not traversable",
              num("clear on truth", n_clear_on_truth) + ", " + num("block", n_block));
        check(hit > 0 && dist >= 0.0 && dist < 6.0, "data pay: cavity localised",
              num("dist m", dist) + ", " + num("hit", hit));
    } else {
        check(hit == 0, "data do not pay: no confident void on truth",
              num("NotThere on truth", hit));
        std::printf("    localisation not claimed (data do not pay); "
                    "Undecided would still block under the policy map\n");
    }

    // ---- empty control -------------------------------------------------------
    std::printf("\n  B. control, no cavity\n");
    const BinaryProblem prob0 = build_binary_problem(m_model, views0, domain, kLambda, kappa);
    const std::vector<float> pv0 = branch_fractions(prob0, posterior, kBranches, n_threads);
    int false_not = 0, clear_where_cavity = 0, block_where_cavity = 0, cavity_slots = 0;
    for (int i = 0; i < prob0.n_vars(); ++i) {
        const Bit b = classify(pv0[i]);
        if (b == Bit::NotThere) ++false_not;
        const int v = prob0.var_voxel[i];
        // Would-be cavity cells (from the with-cavity truth mask over the solid model).
        // In the empty medium those cells are rock in the domain.
        const float x = kAheadLo.x + (v % kAheadN + 0.5f) * kAheadVoxel;
        const float y = kAheadLo.y + ((v / kAheadN) % kAheadN + 0.5f) * kAheadVoxel;
        const float z = kAheadLo.z + (v / (kAheadN * kAheadN) + 0.5f) * kAheadVoxel;
        if (!in_cavity(x, y, z)) continue;
        ++cavity_slots;
        if (traversable(b)) ++clear_where_cavity;
        else ++block_where_cavity;
    }
    std::printf("    confident voids: %d; would-be cavity cells: %d clear, %d block\n",
                false_not, clear_where_cavity, block_where_cavity);
    check(false_not == 0, "control invents no confident void",
          num("false NotThere", false_not));
    check(block_where_cavity == 0 && clear_where_cavity == cavity_slots,
          "control: ahead region is traversable",
          num("clear", clear_where_cavity) + ", " + num("slots", cavity_slots));

    cudaFree(d_cav);
    cudaFree(d_empty);
    std::printf("\n%s\n\n", failures == 0 ? "all checks passed" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
