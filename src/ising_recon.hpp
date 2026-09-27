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
#include <limits>
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
    // Optional per-variable linear term replacing kappa: kappa plus lambda per
    // neighbour frozen as rock (a bond to rock costs lambda iff x = 1).
    std::vector<double> field;

    int n_vars() const { return int(var_voxel.size()); }
    double lin(int i) const { return field.empty() ? kappa : field[i]; }
    int n_rays() const { return int(d.size()); }
};

/// Build the QUBO from binned data, the no-anomaly model (host medium: the
/// surveyed outer shape, solid rock inside) and the domain of unknown voxels.
///
/// a_bv = a_per_metre * ℓ_bv. Default a_per_metre = mu_rock is the void path
/// (ADR-007 §1): x = 1 removes rock attenuation. For a denser body with
/// declared contrast δμ = μ_body − μ_rock > 0 (ADR-006 §1: μ ∝ ρ), pass
/// a_per_metre = −δμ so the same deficit d_b = τ⁰ − t matches the surplus
/// optical depth. Do not invent a scale: δμ comes from the declared density
/// ratio times the engine's μ_rock.
inline BinaryProblem build_binary_problem(const VoxelMedium& m_model,
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
