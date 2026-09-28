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

#include "engine.hpp"
#include "muon_recon.hpp"
#include "op_muon.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace argos {

/// Build the QUBO from binned muon data, the no-anomaly model (host medium: the
/// surveyed outer shape, solid rock inside) and the domain of unknown sites.
/// Since L1 (ADR-017) this is three plug-ins: a grid BitField, the muon sensor's
/// rows, and the Ising prior. The result is identical, array for array, to the
/// pre-L1 builder (check_engine proves it on every build).
///
/// a_bv = a_per_metre * l_bv. Default a_per_metre = mu_rock is the void path
/// (ADR-007 §1): x = 1 removes rock attenuation. For a denser body with
/// declared contrast dmu = mu_body - mu_rock > 0 (ADR-006 §1: mu ~ rho), pass
/// a_per_metre = -dmu so the same deficit d_b = tau0 - t matches the surplus
/// optical depth. Do not invent a scale: dmu comes from the declared density
/// ratio times the engine's mu_rock.
inline BinaryProblem build_binary_problem(const VoxelMedium& m_model,
                                          const std::vector<MuonView>& views,
                                          const std::vector<char>& domain,
                                          double lambda, double kappa,
                                          double min_open = 30.0,
                                          float a_per_metre =
                                              std::numeric_limits<float>::quiet_NaN()) {
    const BitField bits = BitField::grid(m_model.nx, m_model.ny, m_model.nz, domain);
    const OperatorRows muons = muon_rows(m_model, views, bits, min_open, a_per_metre);
    return assemble(bits, {&muons}, IsingPrior{lambda, kappa});
}

/// Energy of a configuration, from scratch (for tests and the oracle).
inline double binary_energy(const BinaryProblem& p, const std::vector<uint8_t>& x) {
    std::vector<double> r(p.n_rays());
    for (int b = 0; b < p.n_rays(); ++b) r[b] = -p.d[b];
    double prior = 0.0;
    for (int i = 0; i < p.n_vars(); ++i) {
        if (x[i])
            for (int k = p.row_ptr[i]; k < p.row_ptr[i + 1]; ++k) r[p.ray[k]] += p.a[k];
        prior += p.lin(i) * x[i];
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

/// ΔE for a proposed flip of variable i, given residuals that match x.
/// Does not flip and does not draw a random. Shared by the annealer and the
/// host local-field check so there is one formula, not two.
///
/// LYTH (ADR-0026) refuses data-dependent branches. The Metropolis accept
/// (dE <= 0 or uniform < exp(-dE/T)) branches on data and on a random draw,
/// so it stays in C++ and must not become a .lyth kernel.
inline double local_field_dE(const BinaryProblem& p, const std::vector<uint8_t>& x,
                             const std::vector<double>& r, int i) {
    const double delta = x[i] ? -1.0 : 1.0;   // +1: rock -> void
    double dE = p.lin(i) * delta;
    for (int k = p.row_ptr[i]; k < p.row_ptr[i + 1]; ++k) {
        const double a = p.a[k] * delta;
        dE += p.w[p.ray[k]] * (r[p.ray[k]] * a + 0.5 * a * a);
    }
    for (int k = p.nbr_ptr[i]; k < p.nbr_ptr[i + 1]; ++k) {
        const bool before = x[p.nbr[k]] != x[i];
        dE += p.lambda * (before ? -1.0 : 1.0);
    }
    return dE;
}

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
            const double dE = local_field_dE(p, x, r, i);
            // Metropolis accept: stays in C++ (data-dependent + random); not LYTH.
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
                                           std::vector<std::vector<uint8_t>>* keep = nullptr,
                                           uint64_t seed_offset = 0) {
    std::vector<std::vector<uint8_t>> branches(n_branches);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < n_threads; ++t)
        pool.emplace_back([&, t] {
            for (int b = int(t); b < n_branches; b += int(n_threads))
                branches[b] = anneal_branch(p, s, uint64_t(b) + 1 + seed_offset);
        });
    for (auto& th : pool) th.join();
    std::vector<float> frac(p.n_vars(), 0.0f);
    for (const auto& x : branches)
        for (int i = 0; i < p.n_vars(); ++i) frac[i] += x[i];
    for (float& f : frac) f /= float(n_branches);
    if (keep) *keep = std::move(branches);
    return frac;
}

/// Freeze every variable that cannot become void in ANY context, by a bound.
///
/// For a flip 0 -> 1 of variable i, with the other variables in any state:
///   data:  dE >= sum_b w_b a_bi (a_bi / 2 - d_b)     (residual r_b >= -d_b,
///          because every other voxel only ADDS deficit)
///   prior: dE >= kappa + lambda * (#rock-frozen neighbours - #free neighbours)
/// If the sum is >= margin nats, the flip has probability <= e^-margin at the
/// posterior temperature: the variable is rock in every branch that matters,
/// and it is fixed there. Freezing a variable turns its bonds into certainties,
/// which tightens its neighbours' bounds: iterate to a fixpoint. The result is
/// a smaller problem over the variables that can still change, with the bonds
/// to frozen rock folded into their linear field. Exact for MAP; for sampling,
/// the error per voxel and sweep is bounded by e^-margin.
struct FreezeResult {
    BinaryProblem reduced;
    std::vector<int> kept;      // reduced variable -> original variable
    int rounds = 0;
};

inline FreezeResult freeze_provable_rock(const BinaryProblem& p, double margin) {
    const int n = p.n_vars();
    std::vector<double> lb_data(n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int k = p.row_ptr[i]; k < p.row_ptr[i + 1]; ++k)
            lb_data[i] += p.w[p.ray[k]] * p.a[k] * (0.5 * p.a[k] - p.d[p.ray[k]]);
    std::vector<uint8_t> frozen(n, 0);
    FreezeResult out;
    for (bool changed = true; changed; ++out.rounds) {
        changed = false;
        for (int i = 0; i < n; ++i) {
            if (frozen[i]) continue;
            int rock = 0, free = 0;
            for (int k = p.nbr_ptr[i]; k < p.nbr_ptr[i + 1]; ++k)
                (frozen[p.nbr[k]] ? rock : free)++;
            const double lb = lb_data[i] + p.lin(i) + p.lambda * (rock - free);
            if (lb >= margin) { frozen[i] = 1; changed = true; }
        }
    }
    std::vector<int> new_index(n, -1);
    for (int i = 0; i < n; ++i)
        if (!frozen[i]) { new_index[i] = int(out.kept.size()); out.kept.push_back(i); }
    BinaryProblem& r = out.reduced;
    r.lambda = p.lambda;
    r.kappa = p.kappa;
    r.d = p.d;
    r.w = p.w;
    r.row_ptr.push_back(0);
    r.nbr_ptr.push_back(0);
    for (int i : out.kept) {
        r.var_voxel.push_back(p.var_voxel[i]);
        for (int k = p.row_ptr[i]; k < p.row_ptr[i + 1]; ++k) {
            r.ray.push_back(p.ray[k]);
            r.a.push_back(p.a[k]);
        }
        r.row_ptr.push_back(int(r.ray.size()));
        int rock = 0;
        for (int k = p.nbr_ptr[i]; k < p.nbr_ptr[i + 1]; ++k) {
            if (frozen[p.nbr[k]]) ++rock;
            else r.nbr.push_back(new_index[p.nbr[k]]);
        }
        r.nbr_ptr.push_back(int(r.nbr.size()));
        r.field.push_back(p.lin(i) + p.lambda * rock);
    }
    r.voxel_var.assign(p.voxel_var.size(), -1);
    for (int j = 0; j < r.n_vars(); ++j) r.voxel_var[r.var_voxel[j]] = j;
    return out;
}

/// The conditional QUBO of a region of interest, everything else fixed.
///
/// With the variables outside R held at x_bar (the branch consensus), the
/// energy over x_R in {0,1}^n is exactly
///     E(x_R) = const + sum_i h_i x_i + sum_{i<j} J_ij x_i x_j
/// from expanding 1/2 w (r_bar + sum a x)^2 (x^2 = x), the linear field, the
/// bonds inside R ([x_i != x_j] = x_i + x_j - 2 x_i x_j) and the bonds to
/// fixed neighbours ([x_i != x_bar] = x_i or 1 - x_i). Small enough (n <= ~24)
/// to enumerate: the exact local posterior, 2^n branches, no sampling.
struct RoiQubo {
    std::vector<int> vars;        // original variable indices, in R's order
    std::vector<double> h;        // n
    std::vector<double> J;        // n*n, upper triangle used
};

inline RoiQubo roi_qubo(const BinaryProblem& p, const std::vector<int>& roi,
                        const std::vector<uint8_t>& x_bar) {
    const int n = int(roi.size());
    RoiQubo q;
    q.vars = roi;
    q.h.assign(n, 0.0);
    q.J.assign(size_t(n) * n, 0.0);
    std::vector<int> pos(p.n_vars(), -1);
    for (int i = 0; i < n; ++i) pos[roi[i]] = i;
    // residuals with R switched off
    std::vector<double> r(p.n_rays());
    for (int b = 0; b < p.n_rays(); ++b) r[b] = -p.d[b];
    for (int v = 0; v < p.n_vars(); ++v)
        if (x_bar[v] && pos[v] < 0)
            for (int k = p.row_ptr[v]; k < p.row_ptr[v + 1]; ++k) r[p.ray[k]] += p.a[k];
    // data: linear and quadratic, through rays shared by R's variables
    std::vector<std::vector<std::pair<int, double>>> by_ray(p.n_rays());
    for (int i = 0; i < n; ++i)
        for (int k = p.row_ptr[roi[i]]; k < p.row_ptr[roi[i] + 1]; ++k) {
            const int b = p.ray[k];
            const double a = p.a[k];
            q.h[i] += p.w[b] * (r[b] * a + 0.5 * a * a);
            by_ray[b].push_back({i, a});
        }
    for (int b = 0; b < p.n_rays(); ++b)
        for (size_t u = 0; u < by_ray[b].size(); ++u)
            for (size_t v = u + 1; v < by_ray[b].size(); ++v) {
                int i = by_ray[b][u].first, j = by_ray[b][v].first;
                if (i > j) std::swap(i, j);
                q.J[size_t(i) * n + j] += p.w[b] * by_ray[b][u].second * by_ray[b][v].second;
            }
    // prior
    for (int i = 0; i < n; ++i) {
        const int vi = roi[i];
        q.h[i] += p.lin(vi);
        for (int k = p.nbr_ptr[vi]; k < p.nbr_ptr[vi + 1]; ++k) {
            const int vj = p.nbr[k];
            if (pos[vj] >= 0) {
                if (pos[vj] > i) {
                    q.h[i] += p.lambda;
                    q.h[pos[vj]] += p.lambda;
                    q.J[size_t(i) * n + pos[vj]] -= 2.0 * p.lambda;
                }
            } else {
                q.h[i] += p.lambda * (x_bar[vj] ? -1.0 : 1.0);
            }
        }
    }
    return q;
}

enum class Bit : uint8_t { Exists = 0, NotThere = 1, Undecided = 2 };

/// The tri-state map (ADR-007 §3): exists, does not exist, or ~~exists~~.
inline Bit classify(float p_void, float lo = 0.1f, float hi = 0.9f) {
    if (p_void >= hi) return Bit::NotThere;
    if (p_void <= lo) return Bit::Exists;
    return Bit::Undecided;
}

}  // namespace argos
