// =============================================================================
// QuBLAR -- capability: denser ore body under rock (synthetic)
// =============================================================================
// Sibling of check_ising. Does not rewrite the Big Void check. Declared inputs
// (not fitted to a score):
//   Host rock density 2.65 g/cm^3 (Lesparre et al. GJI 183, 1348, 2010 / PDG).
//   rho_ore = 2.65 * 1.25 (+25%, midpoint of the owner's 20-30% brief). Not a
//   published ore grade; the contrast is declared.
//   mu_ore = mu_rock * (rho_ore / rho_rock) via ADR-006 §1 (mu ∝ rho).
//   Prior, schedule, classify 0.1/0.9, 3 m rooms, 6 m localisation-when-paid:
//   same frozen constants as check_ising / ising_recon.hpp.
// Exposure is the free dial. Days are not claimed without I and T (see RESULTS).
// =============================================================================

#include "muon.cuh"
#include "muon_recon.hpp"
#include "muon_replica.hpp"
#include "ising_recon.hpp"
#include "run_talk.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace argos;

// Declared scene (not the pyramid replica).
static const int kMineN = 48;
static const float kMineVoxel = 2.0f;
static const float3 kMineLo = make_float3(-48.0f, -48.0f, -4.0f);
static const float kRockHalf = 36.0f;     // |x|,|y| <= this, z in [0, rock_top]
static const float kRockTop = 48.0f;
static const float kOreZ = 28.0f;         // ore body centre (above the chambers)
static const float kRhoRatio = 1.25f;     // declared: rho_ore / rho_rock
static const float kMuOre = kMuRock * kRhoRatio;
static const float kAOre = -(kMuOre - kMuRock);  // surplus optical depth -> deficit

static const double kPriorAnomalyFraction = 1e-3;
static const double kLambda = 2.0;
static const int kBranches = 16;

static const float3 kMineChambers[3] = {
    make_float3(0.0f, 0.0f, 6.0f),
    make_float3(-16.0f, 0.0f, 6.0f),
    make_float3(16.0f, 0.0f, 6.0f),
};

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
                // detector rooms: surveyed air, not inferred
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

struct BodyResult {
    float half = 0;
    int candidates = 0;
    double gain = 0, cost = 0;
    bool data_pay = false;
    int n_ore_label = 0, hit = 0, truth_n = 0, false_ore = 0;
    int u2_miss = 0;  // ADR-019 U2: true-ore bits called rock with confidence
    double dist = -1.0;
    bool ok = false;
};

static BodyResult run_body(float ore_half, int candidates, unsigned n_threads, RunTalk& talk) {
    BodyResult R;
    R.half = ore_half;
    R.candidates = candidates;

    std::vector<float> host_ore, host_empty;
    float *d_ore = nullptr, *d_empty = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ore, size_t(kMineN) * kMineN * kMineN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_empty, size_t(kMineN) * kMineN * kMineN * sizeof(float)));
    const VoxelMedium m_ore = make_mine(true, ore_half, host_ore, d_ore);
    const VoxelMedium m_empty = make_mine(false, ore_half, host_empty, d_empty);
    VoxelMedium m_model = m_empty;
    m_model.mu = host_empty.data();

    std::vector<char> domain;
    fill_domain(host_empty, domain);
    std::vector<char> truth(host_ore.size(), 0);
    for (size_t v = 0; v < truth.size(); ++v)
        truth[v] = domain[v] && host_ore[v] > kMuRock * 1.01f;

    std::vector<MuonBinnedData> seen, seen0;
    {
        Tick gpu(talk, true);
        for (int c = 0; c < 3; ++c) {
            seen.push_back(expose(m_ore, kMineChambers[c], 1u + c, imaging_sky(), candidates));
            seen0.push_back(expose(m_empty, kMineChambers[c], 1u + c, imaging_sky(), candidates));
        }
    }
    const std::vector<MuonView> views = {{kMineChambers[0], &seen[0]},
                                         {kMineChambers[1], &seen[1]},
                                         {kMineChambers[2], &seen[2]}};
    const std::vector<MuonView> views0 = {{kMineChambers[0], &seen0[0]},
                                          {kMineChambers[1], &seen0[1]},
                                          {kMineChambers[2], &seen0[2]}};

    const double kappa = std::log((1.0 - kPriorAnomalyFraction) / kPriorAnomalyFraction);
    Schedule posterior;
    const BinaryProblem prob =
        build_binary_problem(m_model, views, domain, kLambda, kappa, 30.0, kAOre);
    const BinaryProblem prob_d =
        build_binary_problem(m_model, views, domain, 0.0, 0.0, 30.0, kAOre);
    evidence_budget(prob_d, prob, truth, R.gain, R.cost);
    R.data_pay = R.gain > R.cost;

    std::vector<float> pv;
    {
        Tick cpu(talk, false);
        pv = branch_fractions(prob, posterior, kBranches, n_threads);
    }
    double cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < prob.n_vars(); ++i) {
        const int v = prob.var_voxel[i];
        R.truth_n += truth[v];
        R.u2_miss += truth[v] && classify(pv[i]) == Bit::Exists;
        if (classify(pv[i]) != Bit::NotThere) continue;
        ++R.n_ore_label;
        R.hit += truth[v];
        cx += kMineLo.x + (v % kMineN + 0.5f) * kMineVoxel;
        cy += kMineLo.y + ((v / kMineN) % kMineN + 0.5f) * kMineVoxel;
        cz += kMineLo.z + (v / (kMineN * kMineN) + 0.5f) * kMineVoxel;
    }
    if (R.n_ore_label > 0) {
        cx /= R.n_ore_label;
        cy /= R.n_ore_label;
        cz /= R.n_ore_label;
        const double dx = cx, dy = cy, dz = cz - kOreZ;
        R.dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    const BinaryProblem prob0 =
        build_binary_problem(m_model, views0, domain, kLambda, kappa, 30.0, kAOre);
    std::vector<float> pv0;
    {
        Tick cpu(talk, false);
        pv0 = branch_fractions(prob0, posterior, kBranches, n_threads);
    }
    for (float p : pv0)
        if (classify(p) == Bit::NotThere) ++R.false_ore;

    // Exit rules: correct decline or correct paid localisation; never a confident
    // hit when the budget declines; never a confident body on the no-ore control.
    bool ok = (R.false_ore == 0);
    if (R.data_pay)
        ok = ok && R.n_ore_label > 0 && R.dist >= 0.0 && R.dist < 6.0;
    else
        ok = ok && R.hit == 0;
    R.ok = ok;

    cudaFree(d_ore);
    cudaFree(d_empty);
    return R;
}

int main(int argc, char** argv) {
    RunTalk talk = RunTalk::begin("check_mine");
    std::printf("\nQuBLAR -- check_mine (denser ore, declared contrast)\n\n");
    std::printf("  declared: rho_rock = 2.65 g/cm^3 (Lesparre / PDG); "
                "rho_ore/rho_rock = %.2f; mu_ore = %.4g /m; a_per_metre = %.4g /m\n",
                kRhoRatio, kMuOre, kAOre);
    std::printf("  prior p0 = %g, lambda = %.2f, kappa = ln((1-p0)/p0); "
                "classify 0.1/0.9; rooms 3 m; localisation 6 m only if data pay\n",
                kPriorAnomalyFraction, kLambda);
    std::printf("  days are not claimed: exposure is candidate counts only "
                "(Lesparre T and I not derived for this depth from the code flux)\n\n");

    const unsigned n_threads =
        std::max(1u, std::min(16u, std::thread::hardware_concurrency() > 2
                                       ? std::thread::hardware_concurrency() - 2
                                       : 1u));

    // Same schedule for both sizes. Exposure is the free dial (optional argv[1]
    // = log2 candidates); default sweeps two points around the void-path scale.
    const float halves[2] = {2.0f, 4.0f};  // body half-extents: 4 m and 8 m cubes
    std::vector<int> exposures;
    if (argc > 1)
        exposures.push_back(1 << std::atoi(argv[1]));
    else {
        exposures.push_back(1 << 25);
        exposures.push_back(1 << 27);
    }

    for (float half : halves) {
        std::printf("  --- ore half-extent %.1f m (cube ~%.0f m) ---\n", half, 2.0 * half);
        for (int cand : exposures) {
            const auto t0 = std::chrono::steady_clock::now();
            const BodyResult R = run_body(half, cand, n_threads, talk);
            const int log2e = int(std::log2(double(cand)) + 0.5);
            std::printf("    exposure 2^%d = %d candidates/chamber (%.1f s)\n", log2e, cand,
                        seconds_since(t0));
            std::printf("    truth ore voxels: %d; confident ore labels: %d (%d correct); "
                        "control confident ore: %d\n",
                        R.truth_n, R.n_ore_label, R.hit, R.false_ore);
            std::printf("    evidence: data %+.1f nats, prior %+.1f nats -> %s\n", R.gain,
                        R.cost, R.data_pay ? "data pay" : "DECLINE (prior wins)");
            if (R.data_pay)
                std::printf("    localisation: centroid %.2f m from ore centre\n", R.dist);
            else
                std::printf("    localisation not claimed (data do not pay)\n");

            char tag[128];
            std::snprintf(tag, sizeof(tag), "half=%.1f, 2^%d", half, log2e);
            if (R.data_pay)
                check(R.ok, "paid ore localised; control clean",
                      std::string(tag) + ", " + num("dist m", R.dist) + ", "
                          + num("false", R.false_ore));
            else
                check(R.ok, "correct decline; no confident true-ore; control clean",
                      std::string(tag) + ", " + num("hit", R.hit) + ", "
                          + num("false", R.false_ore));
            // ADR-019: U1 is exit-changing, U2 is reported only.
            check(R.n_ore_label == R.hit, "U1 (ADR-019): no confident ore off the truth",
                  std::string(tag) + ", " + num("off truth", R.n_ore_label - R.hit));
            std::printf("    U2 (ADR-019, reported): %d true-ore bits called rock with confidence\n",
                        R.u2_miss);
        }
        std::printf("\n");
    }

    const char* verdict = failures == 0 ? "PASS" : "FAIL";
    std::printf("%s\n\n", failures == 0 ? "all checks passed" : "FAILURES PRESENT");
    talk.end(verdict);
    return failures == 0 ? 0 : 1;
}
