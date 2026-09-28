// =============================================================================
// QuBLAR -- diag_u1_muons: why do muons alone mark confident bits off the ore?
// =============================================================================
// A diagnostic, not a check: it measures the one U1 failure of ADR-019
// (check_fusion, muons only, 3 confident bits off the ore) and changes nothing.
// It feeds the pre-registration of the fix, which must say what was looked at
// before it was frozen. Scene, seeds and prior are check_fusion's, copied
// verbatim; step 0 reproduces the failure, which validates the copy.
//
// Three candidate causes, each with its own test:
//   MODEL    the false bits lower the posterior energy even next to the truth
//            (E(truth + false) < E(truth)): the likelihood and prior prefer them.
//   NOISE    they follow one draw of muon counts: other exposure seeds move them.
//   SAMPLER  16 branches from all-rock share a basin: other annealing seeds do
//            not reproduce them, and E(best branch) > E(truth).
// =============================================================================

#include "muon.cuh"
#include "muon_recon.hpp"
#include "muon_replica.hpp"
#include "ising_recon.hpp"
#include "run_talk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

using namespace argos;

// ---- check_mine's scene, verbatim (as in check_fusion) ------------------------------
static const int kMineN = 48;
static const float kMineVoxel = 2.0f;
static const float3 kMineLo = make_float3(-48.0f, -48.0f, -4.0f);
static const float kRockHalf = 36.0f;
static const float kRockTop = 48.0f;
static const float kOreZ = 28.0f;
static const float kRhoRatio = 1.25f;
static const float kMuOre = kMuRock * kRhoRatio;
static const float kAOre = -(kMuOre - kMuRock);
static const float3 kMineChambers[3] = {
    make_float3(0.0f, 0.0f, 6.0f),
    make_float3(-16.0f, 0.0f, 6.0f),
    make_float3(16.0f, 0.0f, 6.0f),
};

static bool in_rock(float x, float y, float z) {
    return z >= 0.0f && z <= kRockTop && fabsf(x) <= kRockHalf && fabsf(y) <= kRockHalf;
}

static bool in_ore(float x, float y, float z, float half) {
    return fabsf(x) <= half && fabsf(y) <= half && fabsf(z - kOreZ) <= half;
}

static VoxelMedium make_mine(bool with_ore, float ore_half, std::vector<float>& host_mu,
                             float* d_mu) {
    host_mu.assign(size_t(kMineN) * kMineN * kMineN, kMuRock * (1.2f / 2700.0f));
    for (int k = 0; k < kMineN; ++k)
        for (int j = 0; j < kMineN; ++j)
            for (int i = 0; i < kMineN; ++i) {
                const float x = kMineLo.x + (i + 0.5f) * kMineVoxel;
                const float y = kMineLo.y + (j + 0.5f) * kMineVoxel;
                const float z = kMineLo.z + (k + 0.5f) * kMineVoxel;
                const size_t idx = (size_t(k) * kMineN + j) * kMineN + i;
                if (!in_rock(x, y, z)) continue;
                host_mu[idx] = kMuRock;
                if (with_ore && in_ore(x, y, z, ore_half)) host_mu[idx] = kMuOre;
                for (const float3& c : kMineChambers) {
                    const float dx = x - c.x, dy = y - c.y, dz = z - c.z;
                    if (dx * dx + dy * dy + dz * dz < 3.0f * 3.0f)
                        host_mu[idx] = kMuRock * (1.2f / 2700.0f);
                }
            }
    VoxelMedium m;
    m.lo = kMineLo;
    m.voxel = kMineVoxel;
    m.nx = m.ny = m.nz = kMineN;
    m.mu_rock = kMuRock;
    m.mu_air = kMuRock * (1.2f / 2700.0f);
    m.mu = d_mu;
    CUDA_CHECK(cudaMemcpy(d_mu, host_mu.data(), host_mu.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    return m;
}

static void fill_domain(const std::vector<float>& host_solid, std::vector<char>& domain) {
    domain.assign(host_solid.size(), 0);
    for (int k = 0; k < kMineN; ++k)
        for (int j = 0; j < kMineN; ++j)
            for (int i = 0; i < kMineN; ++i) {
                const size_t v = (size_t(k) * kMineN + j) * kMineN + i;
                if (host_solid[v] < kMuRock * 0.5f) continue;
                const float x = kMineLo.x + (i + 0.5f) * kMineVoxel;
                const float y = kMineLo.y + (j + 0.5f) * kMineVoxel;
                const float z = kMineLo.z + (k + 0.5f) * kMineVoxel;
                bool room = false;
                for (const float3& c : kMineChambers) {
                    const float dx = x - c.x, dy = y - c.y, dz = z - c.z;
                    room |= dx * dx + dy * dy + dz * dz < 3.0f * 3.0f;
                }
                domain[v] = room ? 0 : 1;
            }
}

static const float kHalf = 4.0f;
static const int kCandidates = 1 << 23;
static const double kP0 = 1e-3, kLambda = 2.0;
static const int kBranches = 16;

static void centre(int site, double c[3]) {
    c[0] = kMineLo.x + (site % kMineN + 0.5) * kMineVoxel;
    c[1] = kMineLo.y + ((site / kMineN) % kMineN + 0.5) * kMineVoxel;
    c[2] = kMineLo.z + (site / (kMineN * kMineN) + 0.5) * kMineVoxel;
}

/// Confident bits off the truth, as bit indices.
static std::vector<int> off_truth(const std::vector<float>& p, const std::vector<char>& truth) {
    std::vector<int> out;
    for (int b = 0; b < int(p.size()); ++b)
        if (p[b] >= 0.9f && !truth[b]) out.push_back(b);
    return out;
}

/// Per-bit Fisher weight of one sensor, sum over rows of w a^2 (nats per unit x^2).
static std::vector<double> fisher(const OperatorRows& op, int n_bits) {
    std::vector<double> f(n_bits, 0.0);
    for (const auto& e : op.entries) f[e.bit] += op.w[e.row] * double(e.a) * e.a;
    return f;
}

int main() {
    RunTalk talk = RunTalk::begin("diag_u1_muons");
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned threads = hw > 2 ? hw - 2 : 1;

    std::vector<float> host_ore, host_empty;
    float *d_ore = nullptr, *d_empty = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ore, size_t(kMineN) * kMineN * kMineN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_empty, size_t(kMineN) * kMineN * kMineN * sizeof(float)));
    const VoxelMedium m_ore = make_mine(true, kHalf, host_ore, d_ore);
    const VoxelMedium m_empty = make_mine(false, kHalf, host_empty, d_empty);
    VoxelMedium m_model = m_empty;
    m_model.mu = host_empty.data();
    std::vector<char> domain;
    fill_domain(host_empty, domain);
    const BitField bits = BitField::grid(kMineN, kMineN, kMineN, domain);
    const int n = bits.n_bits();
    std::vector<char> truth(n, 0);
    for (int b = 0; b < n; ++b) truth[b] = host_ore[bits.site[b]] > kMuRock * 1.01f;

    const double kappa = std::log((1.0 - kP0) / kP0);
    const IsingPrior prior{kLambda, kappa};
    Schedule posterior;

    auto expose_all = [&](unsigned seed_base) {
        std::vector<MuonBinnedData> seen;
        Tick gpu(talk, true);
        for (int c = 0; c < 3; ++c)
            seen.push_back(expose(m_ore, kMineChambers[c], seed_base + c, imaging_sky(), kCandidates));
        return seen;
    };

    // ---- 0. reproduce check_fusion's muons-only run ------------------------------------
    const std::vector<MuonBinnedData> seen = expose_all(1u);
    const std::vector<MuonView> views = {{kMineChambers[0], &seen[0]}, {kMineChambers[1], &seen[1]},
                                         {kMineChambers[2], &seen[2]}};
    const OperatorRows mu = muon_rows(m_model, views, bits, 30.0, kAOre);
    const BinaryProblem prob = assemble(bits, {&mu}, prior);
    std::vector<std::vector<uint8_t>> kept;
    std::vector<float> p;
    {
        Tick cpu(talk, false);
        p = branch_fractions(prob, posterior, kBranches, threads, &kept);
    }
    const std::vector<int> bad = off_truth(p, truth);
    int conf = 0, hits = 0;
    for (int b = 0; b < n; ++b)
        if (p[b] >= 0.9f) { ++conf; hits += truth[b]; }
    std::printf("\n  0. reproduction: %d confident, %d on the ore, %zu off the truth "
                "(check_fusion: 3, 0, 3) -> %s\n", conf, hits, bad.size(),
                (conf == 3 && hits == 0) ? "reproduced" : "NOT reproduced, stop");
    if (!(conf == 3 && hits == 0)) { talk.end("NOT_REPRODUCED"); return 2; }

    // ---- 1. where they are, and who sees them -----------------------------------------
    std::vector<std::vector<double>> fv;
    for (int c = 0; c < 3; ++c) {
        const OperatorRows one = muon_rows(m_model, {views[c]}, bits, 30.0, kAOre);
        fv.push_back(fisher(one, n));
    }
    // Reference: the Fisher weight a bit of the ore gets from each chamber, averaged.
    double ore_f[3] = {0, 0, 0};
    int ore_n = 0;
    for (int b = 0; b < n; ++b)
        if (truth[b]) { ++ore_n; for (int c = 0; c < 3; ++c) ore_f[c] += fv[c][b]; }
    std::printf("\n  1. the false bits (Fisher weight sum w a^2 per chamber, nats)\n");
    std::printf("     ore bits, mean:                               x=0 %8.3f  x=-16 %8.3f  x=+16 %8.3f\n",
                ore_f[0] / ore_n, ore_f[1] / ore_n, ore_f[2] / ore_n);
    for (int b : bad) {
        double c[3];
        centre(bits.site[b], c);
        std::printf("     bit %6d at (%+5.1f, %+5.1f) depth %4.1f m, p %.3f:  x=0 %8.3f  x=-16 %8.3f  x=+16 %8.3f\n",
                    b, c[0], c[1], kRockTop - c[2], p[b], fv[0][b], fv[1][b], fv[2][b]);
    }

    // ---- 2. MODEL: energies --------------------------------------------------------------
    std::vector<uint8_t> xt(truth.begin(), truth.end());
    const double e_truth = binary_energy(prob, xt);
    double e_best = 1e300, e_mean = 0;
    int best = 0;
    for (size_t k = 0; k < kept.size(); ++k) {
        const double e = binary_energy(prob, kept[k]);
        if (e < e_best) { e_best = e; best = int(k); }
        e_mean += e / kept.size();
    }
    std::vector<uint8_t> xtb = xt;
    for (int b : bad) xtb[b] = 1;
    std::vector<uint8_t> xbc = kept[best];
    for (int b : bad) xbc[b] = 0;
    const double e_truth_plus = binary_energy(prob, xtb);
    const double e_best_clean = binary_energy(prob, xbc);
    int best_ones = 0, best_on_ore = 0;
    for (int b = 0; b < n; ++b) { best_ones += kept[best][b]; best_on_ore += kept[best][b] && truth[b]; }
    std::printf("\n  2. energies (posterior, T = 1)\n");
    std::printf("     E(truth)                  %12.2f\n", e_truth);
    std::printf("     E(best branch)            %12.2f   (%d bits on, %d on the ore)\n", e_best,
                best_ones, best_on_ore);
    std::printf("     E(mean branch)            %12.2f\n", e_mean);
    std::printf("     E(truth + false bits) - E(truth)          %+9.2f   (< 0: the model prefers them next to the truth)\n",
                e_truth_plus - e_truth);
    std::printf("     E(best - false bits) - E(best branch)     %+9.2f   (> 0: the branch needs them)\n",
                e_best_clean - e_best);
    std::printf("     -> %s\n", e_best < e_truth ? "E(best) < E(truth): the model prefers the branches"
                                              : "E(truth) < E(best): the sampler misses the truth");

    // ---- 3. SAMPLER: other annealing seeds, same data ---------------------------------------
    std::printf("\n  3. same data, other annealing seeds (16 branches each)\n");
    for (uint64_t off : {1000ull, 2000ull, 3000ull}) {
        std::vector<float> q;
        {
            Tick cpu(talk, false);
            q = branch_fractions(prob, posterior, kBranches, threads, nullptr, off);
        }
        const std::vector<int> bq = off_truth(q, truth);
        int same = 0;
        for (int b : bq) same += std::count(bad.begin(), bad.end(), b) > 0;
        int cq = 0, hq = 0;
        for (int b = 0; b < n; ++b)
            if (q[b] >= 0.9f) { ++cq; hq += truth[b]; }
        std::printf("     seed offset %4llu: %d confident, %d on the ore, %zu off the truth (%d the same bits)\n",
                    (unsigned long long)off, cq, hq, bq.size(), same);
    }

    // ---- 4. NOISE: other muon draws, same scene ------------------------------------------------
    std::printf("\n  4. other muon exposures (seeds), same scene, same annealing seeds\n");
    for (unsigned sb : {101u, 201u, 301u}) {
        const std::vector<MuonBinnedData> s2 = expose_all(sb);
        const std::vector<MuonView> v2 = {{kMineChambers[0], &s2[0]}, {kMineChambers[1], &s2[1]},
                                          {kMineChambers[2], &s2[2]}};
        const OperatorRows mu2 = muon_rows(m_model, v2, bits, 30.0, kAOre);
        std::vector<float> q;
        {
            Tick cpu(talk, false);
            q = branch_fractions(assemble(bits, {&mu2}, prior), posterior, kBranches, threads);
        }
        const std::vector<int> bq = off_truth(q, truth);
        int cq = 0, hq = 0;
        double zsum = 0;
        for (int b = 0; b < n; ++b)
            if (q[b] >= 0.9f) { ++cq; hq += truth[b]; }
        for (int b : bq) { double c[3]; centre(bits.site[b], c); zsum += kRockTop - c[2]; }
        std::printf("     exposure seeds %3u..%3u: %d confident, %d on the ore, %zu off the truth",
                    sb, sb + 2, cq, hq, bq.size());
        if (!bq.empty()) std::printf(" (mean depth %.1f m)", zsum / bq.size());
        std::printf("\n");
    }

    cudaFree(d_ore);
    cudaFree(d_empty);
    std::printf("\n");
    talk.end("DIAG");
    return 0;
}
