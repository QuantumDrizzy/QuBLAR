// =============================================================================
// QuBLAR -- binary branches: finding what is not there (ADR-007)
// =============================================================================
// Every domain voxel is a bit, x = 1 void / 0 rock. The energy is a QUBO:
//
//   E(x) = 1/2 sum_b w_b (sum_v a_bv x_v - d_b)^2      data (Poisson-weighted)
//        + lambda sum_<uv> [x_u != x_v]                walls are continuous
//        + kappa  sum_v x_v                            voids are rare
//
//   d_b  = tau0_b - t_b      the deficit the data demand on ray b
//   a_bv = mu_rock * l_bv    the deficit voxel v explains if it is void
//   w_b  = N^det_b           1 / Var(-ln T_hat) for T << 1, derived not tuned
//
// With the 1/2 on the data term, E is a negative log-posterior: annealing
// down to temperature 1 and holding there SAMPLES the posterior, so each run
// is one branch drawn with the right weight, and the fraction of branches
// that call a voxel void is its posterior probability. kappa = ln((1-p0)/p0)
// is a prior void fraction p0, declared per experiment; lambda is the Ising
// coupling of the wall prior, also declared.
//
// A flip touches only the rays through that voxel: residuals are kept per
// ray, and each voxel carries its ray list in CSR form.
// =============================================================================

#pragma once

#include "muon_recon.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace argos {

struct BinaryProblem {
    // variables: the domain voxels, in voxel order
    std::vector<int> var_voxel;       // variable -> voxel index
    std::vector<int> voxel_var;       // voxel -> variable, or -1
    // rays: every populated bin of every view
    std::vector<double> d, w;
    // CSR: variable -> (ray, a)
    std::vector<int> row_ptr, ray;
    std::vector<float> a;
    // 6-neighbour adjacency among variables, CSR
    std::vector<int> nbr_ptr, nbr;
    double lambda = 0.0, kappa = 0.0;

    int n_vars() const { return int(var_voxel.size()); }
    int n_rays() const { return int(d.size()); }
};

/// Build the QUBO from binned data, the no-void model (host medium: the
/// surveyed outer shape, solid rock inside) and the domain of unknown voxels.
inline BinaryProblem build_binary_problem(const VoxelMedium& m_model,
                                          const std::vector<MuonView>& views,
                                          const std::vector<char>& domain,
                                          double lambda, double kappa,
                                          double min_open = 30.0) {
    BinaryProblem p;
    p.lambda = lambda;
    p.kappa = kappa;
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
                    entries.push_back({var, r, m_model.mu_rock * seg});
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

/// Energy of a configuration, from scratch (for tests and the oracle).
inline double binary_energy(const BinaryProblem& p, const std::vector<uint8_t>& x) {
    std::vector<double> r(p.n_rays());
    for (int b = 0; b < p.n_rays(); ++b) r[b] = -p.d[b];
    double prior = 0.0;
    for (int i = 0; i < p.n_vars(); ++i) {
        if (x[i])
            for (int k = p.row_ptr[i]; k < p.row_ptr[i + 1]; ++k) r[p.ray[k]] += p.a[k];
        prior += p.kappa * x[i];
        for (int k = p.nbr_ptr[i]; k < p.nbr_ptr[i + 1]; ++k)
            if (p.nbr[k] > i && x[p.nbr[k]] != x[i]) prior += p.lambda;
    }
    double data = 0.0;
    for (int b = 0; b < p.n_rays(); ++b) data += 0.5 * p.w[b] * r[b] * r[b];
    return data + prior;
}

struct Schedule {
    double t_hot = 50.0;    // start: flips of the order of one ray's evidence pass
    double t_cold = 1.0;    // 1 = the posterior; << 1 = ground state (MAP)
    int anneal_sweeps = 200;
    int hold_sweeps = 50;   // sweeps at t_cold before the state is taken
};

namespace detail {
inline uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
inline double uniform(uint64_t& s) { return (splitmix(s) >> 11) * 0x1.0p-53; }
}  // namespace detail

/// One branch: anneal from all-rock at t_hot down to t_cold, hold, return x.
inline std::vector<uint8_t> anneal_branch(const BinaryProblem& p, const Schedule& s,
                                          uint64_t seed) {
    std::vector<uint8_t> x(p.n_vars(), 0);
    std::vector<double> r(p.n_rays());
    for (int b = 0; b < p.n_rays(); ++b) r[b] = -p.d[b];
    uint64_t state = seed * 0x2545F4914F6CDD1Dull + 1;
    const int total = s.anneal_sweeps + s.hold_sweeps;
    for (int sweep = 0; sweep < total; ++sweep) {
        const double frac = std::min(1.0, double(sweep) / std::max(1, s.anneal_sweeps - 1));
        const double temp = sweep < s.anneal_sweeps
            ? s.t_hot * std::pow(s.t_cold / s.t_hot, frac) : s.t_cold;
        for (int i = 0; i < p.n_vars(); ++i) {
            const double delta = x[i] ? -1.0 : 1.0;   // +1: rock -> void
            double dE = p.kappa * delta;
            for (int k = p.row_ptr[i]; k < p.row_ptr[i + 1]; ++k) {
                const double a = p.a[k] * delta;
                dE += p.w[p.ray[k]] * (r[p.ray[k]] * a + 0.5 * a * a);
            }
            for (int k = p.nbr_ptr[i]; k < p.nbr_ptr[i + 1]; ++k) {
                const bool before = x[p.nbr[k]] != x[i];
                dE += p.lambda * (before ? -1.0 : 1.0);
            }
            if (dE <= 0.0 || detail::uniform(state) < std::exp(-dE / temp)) {
                x[i] ^= 1u;
                for (int k = p.row_ptr[i]; k < p.row_ptr[i + 1]; ++k)
                    r[p.ray[k]] += p.a[k] * delta;
            }
        }
    }
    return x;
}

/// R branches in parallel; returns the per-variable void fraction p_v and,
/// optionally, the branches themselves (for export and Blaze).
inline std::vector<float> branch_fractions(const BinaryProblem& p, const Schedule& s,
                                           int n_branches, unsigned n_threads,
                                           std::vector<std::vector<uint8_t>>* keep = nullptr) {
    std::vector<std::vector<uint8_t>> branches(n_branches);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < n_threads; ++t)
        pool.emplace_back([&, t] {
            for (int b = int(t); b < n_branches; b += int(n_threads))
                branches[b] = anneal_branch(p, s, uint64_t(b) + 1);
        });
    for (auto& th : pool) th.join();
    std::vector<float> frac(p.n_vars(), 0.0f);
    for (const auto& x : branches)
        for (int i = 0; i < p.n_vars(); ++i) frac[i] += x[i];
    for (float& f : frac) f /= float(n_branches);
    if (keep) *keep = std::move(branches);
    return frac;
}

enum class Bit : uint8_t { Exists = 0, NotThere = 1, Undecided = 2 };

/// The tri-state map (ADR-007 §3): exists, does not exist, or ~~exists~~.
inline Bit classify(float p_void, float lo = 0.1f, float hi = 0.9f) {
    if (p_void >= hi) return Bit::NotThere;
    if (p_void <= lo) return Bit::Exists;
    return Bit::Undecided;
}

}  // namespace argos
