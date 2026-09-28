// =============================================================================
// QuBLAR -- check_fusion: muons and gravity on one field (ADR-018, pre-registered)
// =============================================================================
// Scene, predictions and exit rules are frozen in ADR-018 (SHA-256 in
// experiments/fusion/PREREG_SHA256.txt) before this file was written. The mine
// scene is check_mine's, copied verbatim (make_mine, fill_domain, in_rock,
// in_ore) so both checks image the same rock.
//
// Three problems on the same bits: muons only, gravity only, fused (both
// sensors' rows through assemble). Each with a body run and a control run.
//   R1 identity: fused data evidence = muon + gravity data evidence (1e-9 rel)
//   R2 no confident bit (p >= 0.9) in any control
//   R3 the question: fused depth error (claimed bits, p >= 0.5) < gravity's
//   R4 if the fused evidence pays: >= 1 confident bit, horizontal centroid < 4 m
// Exit 0 needs R1, R2, R4. R3 is reported as it comes out.
// Bits are drawn as bits: '1' (p >= 0.9), '0' (p <= 0.1), '?' (between).
// =============================================================================

#include "muon.cuh"
#include "muon_recon.hpp"
#include "muon_replica.hpp"
#include "ising_recon.hpp"
#include "op_gravity.hpp"
#include "run_talk.hpp"
#include "ledger_out.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using namespace argos;

// ---- check_mine's scene, verbatim ------------------------------------------------
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

// ---- ADR-018 declared inputs -------------------------------------------------------
static const float kHalf = 4.0f;               // an 8 m cube
static const int kCandidates = 1 << 23;         // per chamber
static const double kSigma = 1.0;               // microGal
static const double kContrast = 0.25 * 2650.0;  // kg/m^3, the same 1.25 ratio
static const double kP0 = 1e-3, kLambda = 2.0;
static const int kBranches = 16;

static double gauss(uint64_t& s) {
    auto u = [&s] {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (double(s >> 11) + 0.5) * 0x1.0p-53;
    };
    const double u1 = u(), u2 = u();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

struct Stats {
    int confident = 0, hits = 0, claimed = 0, undecided = 0;
    int u2_miss = 0;   // ADR-019 U2: ore bits called rock with confidence
    double horiz = -1, depth_claimed = -1, horiz_claimed = -1;
};

static Stats stats(const BitField& bits, const CellGrid& g, const std::vector<char>& truth,
                   const std::vector<float>& p) {
    Stats s;
    double cx = 0, cy = 0, ax = 0, ay = 0, az = 0;
    for (int b = 0; b < bits.n_bits(); ++b) {
        double c[3];
        g.centre(bits.site[b], c);
        if (p[b] > 0.1f && p[b] < 0.9f) ++s.undecided;
        s.u2_miss += truth[b] && p[b] <= 0.1f;
        if (p[b] >= 0.5f) { ++s.claimed; ax += c[0]; ay += c[1]; az += c[2]; }
        if (p[b] >= 0.9f) { ++s.confident; s.hits += truth[b]; cx += c[0]; cy += c[1]; }
    }
    if (s.confident) s.horiz = std::hypot(cx / s.confident, cy / s.confident);
    if (s.claimed) {
        s.horiz_claimed = std::hypot(ax / s.claimed, ay / s.claimed);
        s.depth_claimed = kRockTop - az / s.claimed;
    }
    return s;
}

static int count_confident(const std::vector<float>& p) {
    int n = 0;
    for (float v : p) n += v >= 0.9f;
    return n;
}

int main() {
    RunTalk talk = RunTalk::begin("check_fusion");
    LedgerOut ledger("check_fusion");
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
    std::vector<char> truth(bits.n_bits(), 0);
    for (int b = 0; b < bits.n_bits(); ++b)
        truth[b] = host_ore[bits.site[b]] > kMuRock * 1.01f;
    int truth_n = 0;
    for (char t : truth) truth_n += t;

    // ---- muons -------------------------------------------------------------------
    std::vector<MuonBinnedData> seen, seen0;
    {
        Tick gpu(talk, true);
        for (int c = 0; c < 3; ++c) {
            seen.push_back(expose(m_ore, kMineChambers[c], 1u + c, imaging_sky(), kCandidates));
            seen0.push_back(expose(m_empty, kMineChambers[c], 1u + c, imaging_sky(), kCandidates));
        }
    }
    const std::vector<MuonView> views = {{kMineChambers[0], &seen[0]}, {kMineChambers[1], &seen[1]},
                                         {kMineChambers[2], &seen[2]}};
    const std::vector<MuonView> views0 = {{kMineChambers[0], &seen0[0]},
                                          {kMineChambers[1], &seen0[1]},
                                          {kMineChambers[2], &seen0[2]}};
    const OperatorRows mu = muon_rows(m_model, views, bits, 30.0, kAOre);
    const OperatorRows mu0 = muon_rows(m_model, views0, bits, 30.0, kAOre);

    // ---- gravity -----------------------------------------------------------------
    CellGrid g;
    g.lo[0] = kMineLo.x; g.lo[1] = kMineLo.y; g.lo[2] = kMineLo.z;
    g.h = kMineVoxel; g.nx = g.ny = g.nz = kMineN;
    std::vector<GravityStation> st, st0;
    for (int j = 0; j < 16; ++j)
        for (int i = 0; i < 16; ++i) {
            st.push_back({{-30.0 + 4.0 * i, -30.0 + 4.0 * j, kRockTop + 0.5}, 0.0});
            st0.push_back(st.back());
        }
    {
        const OperatorRows shape = gravity_rows(g, bits, st, kContrast, kSigma);
        std::vector<double> xt(truth.begin(), truth.end());
        const std::vector<double> clean = shape.apply(xt);
        uint64_t seed = 20260928018ull;
        for (size_t s = 0; s < st.size(); ++s) {
            st[s].g_obs = clean[s] + kSigma * gauss(seed);
            st0[s].g_obs = kSigma * gauss(seed);
        }
    }
    const OperatorRows gr = gravity_rows(g, bits, st, kContrast, kSigma);
    const OperatorRows gr0 = gravity_rows(g, bits, st0, kContrast, kSigma);

    // ---- evidence (R1) --------------------------------------------------------------
    const double kappa = std::log((1.0 - kP0) / kP0);
    const IsingPrior prior{kLambda, kappa}, none{0.0, 0.0};
    std::vector<uint8_t> x1(truth.begin(), truth.end()), x0(bits.n_bits(), 0);
    auto gain = [&](const BinaryProblem& pd) {
        return binary_energy(pd, x0) - binary_energy(pd, x1);
    };
    const double g_mu = gain(assemble(bits, {&mu}, none));
    const double g_gr = gain(assemble(bits, {&gr}, none));
    const double g_fu = gain(assemble(bits, {&mu, &gr}, none));
    BinaryProblem prior_only = assemble(bits, {&mu}, prior);
    std::fill(prior_only.w.begin(), prior_only.w.end(), 0.0);
    const double cost = binary_energy(prior_only, x1) - binary_energy(prior_only, x0);

    // ---- posteriors --------------------------------------------------------------
    Schedule posterior;
    std::vector<float> p_mu, p_gr, p_fu, c_mu, c_gr, c_fu;
    {
        Tick cpu(talk, false);
        p_mu = branch_fractions(assemble(bits, {&mu}, prior), posterior, kBranches, threads);
        p_gr = branch_fractions(assemble(bits, {&gr}, prior), posterior, kBranches, threads);
        p_fu = branch_fractions(assemble(bits, {&mu, &gr}, prior), posterior, kBranches, threads);
        c_mu = branch_fractions(assemble(bits, {&mu0}, prior), posterior, kBranches, threads);
        c_gr = branch_fractions(assemble(bits, {&gr0}, prior), posterior, kBranches, threads);
        c_fu = branch_fractions(assemble(bits, {&mu0, &gr0}, prior), posterior, kBranches, threads);
    }
    const Stats s_mu = stats(bits, g, truth, p_mu);
    const Stats s_gr = stats(bits, g, truth, p_gr);
    const Stats s_fu = stats(bits, g, truth, p_fu);
    const double truth_depth = kRockTop - kOreZ;

    std::printf("  %d bits, ore %d bits (8 m cube, %.0f m below the surface)\n", bits.n_bits(),
                truth_n, truth_depth);
    std::printf("  muons 2^23/chamber (%d rows), gravity 256 stations at %.0f microGal, prior %.1f nats\n\n",
                mu.n_rows(), kSigma, cost);
    std::printf("  sensor    data nats  budget   confident (on ore)  claimed  undecided  "
                "horiz(conf)  depth(claimed)  control\n");
    auto row = [&](const char* name, double gn, const Stats& s, int ctrl) {
        std::printf("  %-8s %10.1f  %-7s %9d (%2d)       %6d  %9d   %8.2f m   %8.2f m      %5d\n",
                    name, gn, gn > cost ? "pay" : "DECLINE", s.confident, s.hits, s.claimed,
                    s.undecided, s.horiz, s.depth_claimed, ctrl);
    };
    row("muons", g_mu, s_mu, count_confident(c_mu));
    row("gravity", g_gr, s_gr, count_confident(c_gr));
    row("fused", g_fu, s_fu, count_confident(c_fu));
    std::printf("  (truth depth %.1f m; horizontal truth 0 m)\n\n", truth_depth);

    int fails = 0;
    const bool r1 = std::fabs(g_fu - (g_mu + g_gr)) <= 1e-9 * std::max(1.0, std::fabs(g_fu));
    const bool r2 = count_confident(c_mu) == 0 && count_confident(c_gr) == 0 &&
                    count_confident(c_fu) == 0;
    const bool fused_pays = g_fu > cost;
    const bool r4 = !fused_pays || (s_fu.confident > 0 && s_fu.horiz >= 0 && s_fu.horiz < 4.0);
    const char* r3 = (s_fu.claimed == 0 || s_gr.claimed == 0)
        ? "NOT DECIDABLE"
        : (std::fabs(s_fu.depth_claimed - truth_depth) < std::fabs(s_gr.depth_claimed - truth_depth)
               ? "PASS" : "FAIL");
    std::printf("  R1 fused evidence = muons + gravity: %.6f vs %.6f   %s\n", g_fu, g_mu + g_gr,
                r1 ? "PASS" : "FAIL");
    std::printf("  R2 no confident bit in any control                       %s\n", r2 ? "PASS" : "FAIL");
    std::printf("  R3 fused depth error < gravity's (the question)          %s\n", r3);
    std::printf("  R4 fused pays -> confident bits within 4 m horizontally  %s\n", r4 ? "PASS" : "FAIL");
    fails += !r1 + !r2 + !r4;

    // ADR-019: U1 (exit-changing) per sensor, U2 reported only.
    const struct { const char* name; const Stats* s; } u[3] = {
        {"muons", &s_mu}, {"gravity", &s_gr}, {"fused", &s_fu}};
    for (const auto& r : u) {
        const bool u1 = r.s->confident == r.s->hits;
        fails += !u1;
        std::printf("  U1 (ADR-019) %-8s %3d confident off the truth          %s   "
                    "U2 (reported): %d ore bits called rock\n",
                    r.name, r.s->confident - r.s->hits, u1 ? "PASS" : "FAIL", r.s->u2_miss);
    }

    // ---- the bits: a section through the ore, y = +1 m -------------------------------
    std::printf("\n  section y = +1 m (x from -36 to +36 m); '1' p>=0.9, '0' p<=0.1, '?' between\n");
    std::printf("   depth  truth                                 muons                                 "
                "gravity                               fused\n");
    const int j = 24;
    auto cell = [&](const std::vector<float>& p, int b) {
        return p[b] >= 0.9f ? '1' : (p[b] <= 0.1f ? '0' : '?');
    };
    for (int k = kMineN - 1; k >= 0; --k) {
        const float z = kMineLo.z + (k + 0.5f) * kMineVoxel;
        if (z < 12.0f || z > kRockTop) continue;
        std::string t, a, b2, c;
        for (int i = 6; i < 42; ++i) {
            const int b = bits.site_bit[(size_t(k) * kMineN + j) * kMineN + i];
            if (b < 0) { t += ' '; a += ' '; b2 += ' '; c += ' '; continue; }
            t += truth[b] ? '1' : '0';
            a += cell(p_mu, b);
            b2 += cell(p_gr, b);
            c += cell(p_fu, b);
        }
        std::printf("  %4.0f m  %s  %s  %s  %s\n", kRockTop - z, t.c_str(), a.c_str(), b2.c_str(),
                    c.c_str());
    }

    cudaFree(d_ore);
    cudaFree(d_empty);
    std::printf("\n  %s\n", fails ? "FAILURES" : "all exit rules passed");

    // ADR-021: every sensor and every rule, for the chain. The muons-only U1
    // failure is registered (ADR-019 section 4) and is signed as a FAIL.
    {
        const std::string in = "{\"muon_candidates_log2\":23,\"sigma_ugal\":1,\"ore_half_m\":4,\"ore_depth_m\":20}";
        const char* src = "check_fusion: muons + gravity on one field (ADR-018, ADR-019)";
        const struct { const char* key; double g; const Stats* s; int ctrl; } rows[3] = {
            {"muons", g_mu, &s_mu, count_confident(c_mu)},
            {"gravity", g_gr, &s_gr, count_confident(c_gr)},
            {"fused", g_fu, &s_fu, count_confident(c_fu)}};
        for (const auto& r : rows) {
            const std::string k = std::string(r.key) + "/";
            const bool pays = r.g > cost;
            ledger.num(k + "data_nats", r.g, "nats", pays ? "PAY" : "DECLINE", src, in);
            ledger.num(k + "confident", r.s->confident, "bits", "REPORTED", src, in);
            ledger.num(k + "on_ore", r.s->hits, "bits", "REPORTED", src, in);
            ledger.num(k + "off_truth", r.s->confident - r.s->hits, "bits",
                       r.s->confident == r.s->hits ? "PASS" : "FAIL", "ADR-019 U1: no confident bit off the truth", in);
            ledger.num(k + "u2_misses", r.s->u2_miss, "bits", "REPORTED", "ADR-019 U2 (reported)", in);
            ledger.num(k + "undecided", r.s->undecided, "bits", "REPORTED", src, in);
            ledger.num(k + "control_false", r.ctrl, "bits", r.ctrl == 0 ? "PASS" : "FAIL", src, in);
            if (r.s->claimed) ledger.num(k + "claimed_depth_m", r.s->depth_claimed, "m", "REPORTED", src, in);
            if (r.s->confident) ledger.num(k + "horizontal_m", r.s->horiz, "m", "REPORTED", src, in);
        }
        ledger.num("prior_nats", cost, "nats", "REPORTED", src, in);
        ledger.flag("r1_evidence_adds", r1, r1 ? "PASS" : "FAIL", "ADR-018 R1", in);
        ledger.flag("r2_controls_clean", r2, r2 ? "PASS" : "FAIL", "ADR-018 R2", in);
        ledger.flag("r4_fused_localised", r4, r4 ? "PASS" : "FAIL", "ADR-018 R4", in);
        ledger.flag("r3_decidable", std::string(r3) != "NOT DECIDABLE", "REPORTED", "ADR-018 R3", in);
    }
    ledger.write();
    talk.end(fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
