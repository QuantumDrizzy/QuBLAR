// =============================================================================
// QuBLAR -- check_engine: the L1 model layer (ADR-017)
// =============================================================================
//   A. regression: the L1 assembly builds the SAME problem, array for array,
//      as the pre-L1 builder (frozen verbatim below), on the pyramid, for the
//      void path and the denser-body path
//   B. adjoint: <A x, r> = <x, A^T r> for the muon and the gravity sensors
//   C. gravity physics: a uniform sphere of cells attracts, outside it, as a
//      point mass at its centre (Newton's shell theorem), within 1 %
//   D. fusion: muons and gravity stacked on one field; the energy splits
//      exactly into each sensor's data term plus one prior
// Exit 0 only if all pass.
// =============================================================================

#include "muon.cuh"
#include "muon_recon.hpp"
#include "muon_replica.hpp"
#include "ising_recon.hpp"
#include "op_gravity.hpp"
#include "run_talk.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace argos;

static int failures = 0;

static void check(bool ok, const char* what, const std::string& detail = "") {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", what, detail.empty() ? "" : " -- ",
                detail.c_str());
    if (!ok) ++failures;
}

// ---- the pre-L1 builder, frozen verbatim (only renamed) ----------------------
static BinaryProblem legacy_build_binary_problem(const VoxelMedium& m_model,
                                          const std::vector<MuonView>& views,
                                          const std::vector<char>& domain,
                                          double lambda, double kappa,
                                          double min_open = 30.0,
                                          float a_per_metre =
                                              std::numeric_limits<float>::quiet_NaN()) {
    BinaryProblem p;
    p.lambda = lambda;
    p.kappa = kappa;
    // NaN => void path (a = mu_rock). Ore uses a negative coefficient; do not
    // treat sign as the default sentinel.
    const float a_scale =
        std::isnan(a_per_metre) ? m_model.mu_rock : a_per_metre;
    const size_t n_vox = size_t(m_model.nx) * m_model.ny * m_model.nz;
    p.voxel_var.assign(n_vox, -1);
    for (size_t v = 0; v < n_vox; ++v)
        if (domain[v]) {
            p.voxel_var[v] = int(p.var_voxel.size());
            p.var_voxel.push_back(int(v));
        }

    struct Entry { int var, ray; float a; };
    std::vector<Entry> entries;
    for (const MuonView& view : views) {
        const MuonBinnedData& data = *view.data;
        for (int b = 0; b < data.size(); ++b) {
            if (double(data.open[b]) < min_open) continue;
            const float3 dir = data.bin_direction(b);
            const double tau0 = march_medium(m_model, view.chamber, dir);
            const double det = std::max(double(data.det[b]), 1.0);
            const double t = -std::log(det / double(data.open[b]));
            const int r = p.n_rays();
            bool touches = false;
            march_impl(m_model, view.chamber, dir, [&](int idx, float seg) {
                const int var = p.voxel_var[idx];
                if (var >= 0) {
                    entries.push_back({var, r, a_scale * seg});
                    touches = true;
                }
                return 0.0f;
            });
            if (!touches) continue;
            p.d.push_back(tau0 - t);
            p.w.push_back(det);
        }
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry& x, const Entry& y) { return x.var < y.var; });
    p.row_ptr.assign(p.n_vars() + 1, 0);
    for (const Entry& e : entries) p.row_ptr[e.var + 1]++;
    for (int i = 0; i < p.n_vars(); ++i) p.row_ptr[i + 1] += p.row_ptr[i];
    p.ray.resize(entries.size());
    p.a.resize(entries.size());
    for (size_t k = 0; k < entries.size(); ++k) {
        p.ray[k] = entries[k].ray;
        p.a[k] = entries[k].a;
    }

    const int nx = m_model.nx, ny = m_model.ny, nz = m_model.nz;
    p.nbr_ptr.assign(p.n_vars() + 1, 0);
    for (int i = 0; i < p.n_vars(); ++i) {
        const int v = p.var_voxel[i];
        const int x = v % nx, y = (v / nx) % ny, z = v / (nx * ny);
        const int cand[6][3] = {{x - 1, y, z}, {x + 1, y, z}, {x, y - 1, z},
                                {x, y + 1, z}, {x, y, z - 1}, {x, y, z + 1}};
        for (const auto& c : cand) {
            if (c[0] < 0 || c[0] >= nx || c[1] < 0 || c[1] >= ny || c[2] < 0 || c[2] >= nz)
                continue;
            const int u = p.voxel_var[(size_t(c[2]) * ny + c[1]) * nx + c[0]];
            if (u >= 0) p.nbr.push_back(u);
        }
        p.nbr_ptr[i + 1] = int(p.nbr.size());
    }
    return p;
}

// -----------------------------------------------------------------------------

template <class T>
static bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

static bool identical(const BinaryProblem& x, const BinaryProblem& y) {
    return same(x.var_voxel, y.var_voxel) && same(x.voxel_var, y.voxel_var) &&
           same(x.d, y.d) && same(x.w, y.w) && same(x.row_ptr, y.row_ptr) &&
           same(x.ray, y.ray) && same(x.a, y.a) && same(x.nbr_ptr, y.nbr_ptr) &&
           same(x.nbr, y.nbr) && x.lambda == y.lambda && x.kappa == y.kappa &&
           same(x.field, y.field);
}

static double prior_energy(const BinaryProblem& p, const std::vector<uint8_t>& x) {
    BinaryProblem q = p;
    std::fill(q.w.begin(), q.w.end(), 0.0);
    return binary_energy(q, x);
}

int main() {
    RunTalk talk = RunTalk::begin("check_engine");
    const int cands = 1 << 20;   // small on purpose: the arrays, not the image, are tested

    std::vector<float> host_mu, host_mu_empty;
    float *d_mu = nullptr, *d_mu_empty = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mu, size_t(kN) * kN * kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mu_empty, size_t(kN) * kN * kN * sizeof(float)));
    const VoxelMedium m_void = make_medium(true, host_mu, d_mu);
    const VoxelMedium m_empty = make_medium(false, host_mu_empty, d_mu_empty);
    VoxelMedium m_model = m_empty;
    m_model.mu = host_mu_empty.data();
    std::vector<char> domain(host_mu_empty.size(), 0);
    for (size_t v = 0; v < domain.size(); ++v) domain[v] = host_mu_empty[v] >= kMuRock;

    std::vector<MuonBinnedData> seen;
    {
        Tick gpu(talk, true);
        for (int c = 0; c < 3; ++c)
            seen.push_back(expose(m_void, kChambers[c], 1u + c, imaging_sky(), cands));
    }
    const std::vector<MuonView> views = {{kChambers[0], &seen[0]}, {kChambers[1], &seen[1]},
                                         {kChambers[2], &seen[2]}};

    // ---- A. regression ---------------------------------------------------------
    std::printf("\n  A. the L1 assembly against the frozen pre-L1 builder\n");
    {
        const BinaryProblem legacy = legacy_build_binary_problem(m_model, views, domain, 2.0, 6.9);
        const BinaryProblem now = build_binary_problem(m_model, views, domain, 2.0, 6.9);
        check(identical(legacy, now), "void path: identical problem, array for array",
              std::to_string(now.n_vars()) + " bits, " + std::to_string(now.n_rays()) +
                  " rows, " + std::to_string(now.a.size()) + " entries");
        const float ore = -0.0115f;
        const BinaryProblem legacy_o =
            legacy_build_binary_problem(m_model, views, domain, 2.0, 6.9, 30.0, ore);
        const BinaryProblem now_o = build_binary_problem(m_model, views, domain, 2.0, 6.9, 30.0, ore);
        check(identical(legacy_o, now_o), "denser-body path: identical problem, array for array");
    }

    // ---- B. adjoint --------------------------------------------------------------
    std::printf("\n  B. adjoint test of each sensor\n");
    const BitField bits = BitField::grid(kN, kN, kN, domain);
    const OperatorRows muons = muon_rows(m_model, views, bits);
    CellGrid grid;
    grid.lo[0] = kLo.x; grid.lo[1] = kLo.y; grid.lo[2] = kLo.z;
    grid.h = kVoxel; grid.nx = grid.ny = grid.nz = kN;
    std::vector<GravityStation> stations;
    for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i)
            stations.push_back({{-70.0 + 20.0 * i, -70.0 + 20.0 * j, 160.0}, 0.0});
    const OperatorRows grav = gravity_rows(grid, bits, stations, 662.5, 5.0);
    {
        const double em = adjoint_mismatch(muons, bits.n_bits());
        const double eg = adjoint_mismatch(grav, bits.n_bits());
        char buf[96];
        std::snprintf(buf, sizeof buf, "relative mismatch %.2e", em);
        check(em < 1e-12, "muon sensor: <Ax, r> = <x, A^T r>", buf);
        std::snprintf(buf, sizeof buf, "relative mismatch %.2e", eg);
        check(eg < 1e-12, "gravity sensor: <Ax, r> = <x, A^T r>", buf);
    }

    // ---- C. gravity against the shell theorem -----------------------------------
    std::printf("\n  C. gravity: a sphere of cells against a point mass at its centre\n");
    {
        CellGrid g;
        g.lo[0] = g.lo[1] = -16.0; g.lo[2] = -40.0; g.h = 1.0; g.nx = g.ny = 32; g.nz = 40;
        std::vector<char> all(size_t(g.nx) * g.ny * g.nz, 1);
        const BitField f = BitField::grid(g.nx, g.ny, g.nz, all);
        const double cz = -20.0, R = 6.0, drho = 662.5;
        const std::vector<GravityStation> st = {{{3.0, -2.0, 0.5}, 0.0}, {{0.0, 0.0, 0.5}, 0.0},
                                               {{-9.0, 7.0, 0.5}, 0.0}};
        const OperatorRows op = gravity_rows(g, f, st, drho, 5.0);
        std::vector<double> x(f.n_bits(), 0.0);
        int inside = 0;
        for (int b = 0; b < f.n_bits(); ++b) {
            double c[3];
            g.centre(f.site[b], c);
            if (c[0] * c[0] + c[1] * c[1] + (c[2] - cz) * (c[2] - cz) <= R * R) {
                x[b] = 1.0;
                ++inside;
            }
        }
        const std::vector<double> y = op.apply(x);
        double worst = 0.0;
        const double centre[3] = {0.0, 0.0, cz};
        for (size_t s = 0; s < st.size(); ++s) {
            const double exact = point_mass_gz(st[s].p, centre, inside * drho);
            worst = std::max(worst, std::fabs(y[s] - exact) / exact);
        }
        char buf[128];
        std::snprintf(buf, sizeof buf, "%d cells, worst relative error %.3f %% over 3 stations",
                      inside, 100.0 * worst);
        check(worst < 0.01, "sphere of cells = point mass at its centre, within 1 %", buf);
    }

    // ---- D. fusion ---------------------------------------------------------------
    std::printf("\n  D. two sensors on one field\n");
    {
        const IsingPrior prior{2.0, 6.9};
        const BinaryProblem pm = assemble(bits, {&muons}, prior);
        const BinaryProblem pg = assemble(bits, {&grav}, prior);
        const BinaryProblem pf = assemble(bits, {&muons, &grav}, prior);
        check(pf.n_rays() == pm.n_rays() + pg.n_rays() && pf.a.size() == pm.a.size() + pg.a.size(),
              "rows and entries stack",
              std::to_string(pm.n_rays()) + " muon + " + std::to_string(pg.n_rays()) +
                  " gravity rows");
        uint64_t s = 11;
        std::vector<uint8_t> x(bits.n_bits());
        for (auto& v : x) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            v = uint8_t((s >> 60) == 0);   // about 1 bit in 16 set
        }
        const double ef = binary_energy(pf, x);
        const double split = binary_energy(pm, x) + binary_energy(pg, x) - prior_energy(pm, x);
        char buf[96];
        std::snprintf(buf, sizeof buf, "fused %.6f vs split %.6f", ef, split);
        check(std::fabs(ef - split) <= 1e-9 * std::fabs(ef), "E(fused) = E(muons) + E(gravity) - prior",
              buf);
    }

    cudaFree(d_mu);
    cudaFree(d_mu_empty);
    std::printf("\n  %s\n", failures ? "FAILURES" : "all checks passed");
    talk.end(failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
