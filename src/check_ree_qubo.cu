// =============================================================================
// QuBLAR -- ADR-013: rare-earth SX separation sequencing as a QUBO
// =============================================================================
// Builds the sequencing QUBO (ADR-013 sec. 3), validates it by brute force, solves it with the
// repo Ising engine (argos::anneal_branch, unchanged, Gset MAP schedule) through a "pair-ray"
// encoding, and computes the exact optimum (DP + tree enumeration) and greedy baselines.
// Writes QUBO files for the neal comparison (tools/ree_neal.py).
// Usage: check_ree_qubo <outdir> [runs=64] [anneal_sweeps=400] [hold_sweeps=20]   (non-default sweeps = exploratory)
// =============================================================================
#include "ising_recon.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace argos;

namespace ree {

constexpr double kPurity = 0.999;

struct Instance {
    std::string id, set;
    int n;
    std::vector<std::string> names;
    std::vector<double> z;       // mole fractions, extractability order
    std::vector<double> beta;    // beta[s] = beta(s+1 / s), s = 0..n-2
};

inline int stages(double beta) {
    const double nmin = std::log(std::pow(kPurity / (1 - kPurity), 2)) / std::log(beta);
    return static_cast<int>(std::ceil(2.0 * nmin));
}

struct Var { int a, s, b; };

struct Model {
    Instance in;
    std::vector<Var> vars;
    std::map<std::tuple<int, int, int>, int> idx;
    std::vector<double> F;           // F[a*n+b]
    std::vector<double> cost;        // per var
    std::vector<int> N;              // stages per split position s
    // constraints: list of (coef, var) + rhs
    struct Con { std::vector<std::pair<int, int>> terms; int rhs; };
    std::vector<Con> cons;
    double P = 0;
    // normalised QUBO: E = offset + sum_i Qd[i] x_i + sum_{i<j} Qo[(i,j)] x_i x_j
    double offset = 0;
    std::vector<double> Qd;
    std::map<std::pair<int, int>, double> Qo;
};

inline Model build(const Instance& in) {
    Model m; m.in = in; const int n = in.n;
    m.F.assign(n * n, 0.0);
    for (int a = 0; a < n; ++a) { double s = 0; for (int b = a; b < n; ++b) { s += in.z[b]; m.F[a * n + b] = s; } }
    for (int s = 0; s + 1 < n; ++s) m.N.push_back(stages(in.beta[s]));
    for (int a = 0; a < n; ++a) for (int b = a + 1; b < n; ++b) for (int s = a; s < b; ++s) {
        m.idx[{a, s, b}] = static_cast<int>(m.vars.size());
        m.vars.push_back({a, s, b});
        m.cost.push_back(m.F[a * n + b] * m.N[s]);
    }
    for (int a = 0; a < n; ++a) for (int b = a + 1; b < n; ++b) {
        Model::Con c; c.rhs = (a == 0 && b == n - 1) ? 1 : 0;
        for (int s = a; s < b; ++s) c.terms.push_back({+1, m.idx[{a, s, b}]});
        if (!(a == 0 && b == n - 1)) {
            for (int cc = b + 1; cc < n; ++cc) c.terms.push_back({-1, m.idx[{a, b, cc}]});     // left child of [a..cc] split at b
            for (int cc = 0; cc < a; ++cc) c.terms.push_back({-1, m.idx[{cc, a - 1, b}]});    // right child of [cc..b] split at a-1
        }
        m.cons.push_back(c);
    }
    return m;
}

// ---- exact: DP + enumeration ----
inline double dp_opt(const Model& m, std::vector<int>* tree) {
    const int n = m.in.n;
    std::vector<double> best(n * n, 0.0); std::vector<int> arg(n * n, -1);
    for (int len = 2; len <= n; ++len) for (int a = 0; a + len - 1 < n; ++a) {
        const int b = a + len - 1; double bv = 1e300;
        for (int s = a; s < b; ++s) {
            const double v = m.F[a * n + b] * m.N[s] + best[a * n + s] + best[(s + 1) * n + b];
            if (v < bv) { bv = v; arg[a * n + b] = s; }
        }
        best[a * n + b] = bv;
    }
    if (tree) {
        tree->clear();
        std::function<void(int, int)> rec = [&](int a, int b) {
            if (b <= a) return; const int s = arg[a * n + b];
            tree->push_back(m.idx.at({a, s, b})); rec(a, s); rec(s + 1, b);
        };
        rec(0, n - 1);
    }
    return best[n - 1];
}

inline void enum_trees(const Model& m, int a, int b, std::vector<std::vector<int>>& out) {
    out.clear();
    if (b <= a) { out.push_back({}); return; }
    for (int s = a; s < b; ++s) {
        std::vector<std::vector<int>> L, R; enum_trees(m, a, s, L); enum_trees(m, s + 1, b, R);
        for (auto& l : L) for (auto& r : R) {
            std::vector<int> t{m.idx.at({a, s, b})}; t.insert(t.end(), l.begin(), l.end()); t.insert(t.end(), r.begin(), r.end());
            out.push_back(t);
        }
    }
}

inline double tree_cost(const Model& m, const std::vector<int>& t) { double c = 0; for (int v : t) c += m.cost[v]; return c; }

// ---- greedy ----
inline double greedy(const Model& m, int mode, std::vector<int>* tree) {
    const int n = m.in.n; std::vector<int> t;
    std::function<void(int, int)> rec = [&](int a, int b) {
        if (b <= a) return;
        int bs = a; double bv = 1e300;
        for (int s = a; s < b; ++s) {
            const double v = mode == 1 ? -m.in.beta[s] : std::fabs(m.F[a * n + s] - m.F[(s + 1) * n + b]);
            if (v < bv) { bv = v; bs = s; }
        }
        t.push_back(m.idx.at({a, bs, b})); rec(a, bs); rec(bs + 1, b);
    };
    rec(0, n - 1);
    if (tree) *tree = t;
    return tree_cost(m, t);
}

inline void make_qubo(Model& m, double greedy_cost) {
    m.P = 1.1 * greedy_cost;
    const int nv = static_cast<int>(m.vars.size());
    std::vector<double> Qd(nv, 0.0); std::map<std::pair<int, int>, double> Qo; double off = 0;
    for (int i = 0; i < nv; ++i) Qd[i] += m.cost[i];
    for (const auto& c : m.cons) {
        // P (sum c_k x_k - r)^2 = P [sum c_k^2 x_k + 2 sum_{k<l} c_k c_l x_k x_l - 2 r sum c_k x_k + r^2]
        for (size_t k = 0; k < c.terms.size(); ++k) {
            const auto [ck, vk] = c.terms[k];
            Qd[vk] += m.P * (ck * ck - 2.0 * c.rhs * ck);
            for (size_t l = k + 1; l < c.terms.size(); ++l) {
                const auto [cl, vl] = c.terms[l];
                const auto key = std::make_pair(std::min(vk, vl), std::max(vk, vl));
                Qo[key] += m.P * 2.0 * ck * cl;
            }
        }
        off += m.P * c.rhs * c.rhs;
    }
    m.offset = off / m.P; m.Qd.resize(nv);
    for (int i = 0; i < nv; ++i) m.Qd[i] = Qd[i] / m.P;
    for (auto& [k, v] : Qo) if (v != 0.0) m.Qo[k] = v / m.P;
}

inline double qubo_energy(const Model& m, const std::vector<uint8_t>& x) {
    double e = m.offset;
    for (size_t i = 0; i < x.size(); ++i) if (x[i]) e += m.Qd[i];
    for (const auto& [k, v] : m.Qo) if (x[k.first] && x[k.second]) e += v;
    return e;
}
inline double penalty_units(const Model& m, const std::vector<uint8_t>& x) {   // sum of squared violations (integer)
    double p = 0;
    for (const auto& c : m.cons) { int s = -c.rhs; for (auto [ck, vk] : c.terms) s += ck * x[vk]; p += double(s) * s; }
    return p;
}
inline double x_cost(const Model& m, const std::vector<uint8_t>& x) { double c = 0; for (size_t i = 0; i < x.size(); ++i) if (x[i]) c += m.cost[i]; return c; }

// pair-ray encoding into the engine's BinaryProblem
inline BinaryProblem to_engine(const Model& m) {
    BinaryProblem p; const int nv = static_cast<int>(m.vars.size());
    p.var_voxel.resize(nv); for (int i = 0; i < nv; ++i) p.var_voxel[i] = i;
    p.voxel_var = p.var_voxel;
    std::vector<std::vector<std::pair<int, float>>> rows(nv);
    std::vector<double> field(m.Qd.begin(), m.Qd.end());
    for (const auto& [k, q] : m.Qo) {
        const int r = p.n_rays(); p.d.push_back(0.0); p.w.push_back(1.0);
        const double sq = std::sqrt(std::fabs(q));
        rows[k.first].push_back({r, static_cast<float>(sq)});
        rows[k.second].push_back({r, static_cast<float>(q >= 0 ? sq : -sq)});
        // the ray adds |q|/2 to each diagonal (with the float-rounded a^2); remove it exactly
        field[k.first] -= 0.5 * double(static_cast<float>(sq)) * double(static_cast<float>(sq));
        field[k.second] -= 0.5 * double(static_cast<float>(sq)) * double(static_cast<float>(sq));
    }
    p.row_ptr.assign(nv + 1, 0);
    for (int i = 0; i < nv; ++i) { p.row_ptr[i + 1] = p.row_ptr[i] + static_cast<int>(rows[i].size());
        for (auto& e : rows[i]) { p.ray.push_back(e.first); p.a.push_back(e.second); } }
    p.nbr_ptr.assign(nv + 1, 0); p.lambda = 0.0; p.kappa = 0.0; p.field = field;
    return p;
}

// instrumented copy of anneal_branch (identical arithmetic) that records the energy after each sweep
inline std::vector<double> anneal_trace(const BinaryProblem& p, const Schedule& s, uint64_t seed) {
    std::vector<uint8_t> x(p.n_vars(), 0);
    std::vector<double> r(p.n_rays());
    for (int b = 0; b < p.n_rays(); ++b) r[b] = -p.d[b];
    uint64_t state = seed * 0x2545F4914F6CDD1Dull + 1;
    const int total = s.anneal_sweeps + s.hold_sweeps;
    std::vector<double> tr;
    for (int sweep = 0; sweep < total; ++sweep) {
        const double frac = std::min(1.0, double(sweep) / std::max(1, s.anneal_sweeps - 1));
        const double temp = sweep < s.anneal_sweeps ? s.t_hot * std::pow(s.t_cold / s.t_hot, frac) : s.t_cold;
        for (int i = 0; i < p.n_vars(); ++i) {
            const double delta = x[i] ? -1.0 : 1.0;
            const double dE = local_field_dE(p, x, r, i);
            if (dE <= 0.0 || detail::uniform(state) < std::exp(-dE / temp)) {
                x[i] ^= 1u;
                for (int k = p.row_ptr[i]; k < p.row_ptr[i + 1]; ++k) r[p.ray[k]] += p.a[k] * delta;
            }
        }
        tr.push_back(binary_energy(p, x));
    }
    return tr;
}

}  // namespace ree

using namespace ree;

static Instance mountain_pass(const std::string& set) {
    Instance in; in.set = set; in.n = 5; in.id = "MP_" + set;
    in.names = {"La", "Ce", "Pr", "Nd", "Sm+"};
    // Haxel (USGS OFR 2005-1219) Table 4, ug/g; Sm+ = Sm+Eu+Gd+Tb+Ho+Tm+Yb+Lu
    const double ugg[5] = {29200, 52800, 3900, 11500, 643 + 114 + 220 + 14.3 + 4.34 + 0.918 + 4.76 + 0.673};
    const double aw[5] = {138.905, 140.116, 140.908, 144.242, 150.36};
    double tot = 0; for (int i = 0; i < 5; ++i) { in.z.push_back(ugg[i] / aw[i]); tot += in.z.back(); }
    for (auto& v : in.z) v /= tot;
    if (set == "P507-A") in.beta = {8.0, 1.84, 1.50, 10.0};
    else in.beta = {8.0, 2.2, 1.6, 10.0};
    return in;
}

static Instance random_instance(int n, int k) {
    Instance in; in.n = n; in.set = "random"; in.id = "R" + std::to_string(n) + "_" + std::to_string(k);
    std::mt19937_64 rng(1000ull * n + k);
    std::gamma_distribution<double> G(1.0, 1.0); std::uniform_real_distribution<double> U(1.4, 2.5);
    double tot = 0; for (int i = 0; i < n; ++i) { in.z.push_back(G(rng)); tot += in.z.back(); in.names.push_back("c" + std::to_string(i)); }
    for (auto& v : in.z) v /= tot;
    for (int s = 0; s + 1 < n; ++s) in.beta.push_back(U(rng));
    return in;
}

int main(int argc, char** argv) {
    const std::string out = argc > 1 ? argv[1] : "build/ree";
    const int runs = argc > 2 ? std::atoi(argv[2]) : 64;
    Schedule map; map.t_hot = 5.0; map.t_cold = 1e-3; map.anneal_sweeps = 400; map.hold_sweeps = 20;   // Gset MAP schedule
    if (argc > 3) map.anneal_sweeps = std::atoi(argv[3]);
    if (argc > 4) map.hold_sweeps = std::atoi(argv[4]);
    std::printf("schedule t_hot %.1f t_cold %.0e anneal %d hold %d runs %d\n", map.t_hot, map.t_cold, map.anneal_sweeps, map.hold_sweeps, runs);

    std::vector<Instance> insts = {mountain_pass("P507-A"), mountain_pass("P507-B")};
    for (int n = 4; n <= 12; ++n) for (int k = 0; k < 10; ++k) insts.push_back(random_instance(n, k));

    FILE* fi = std::fopen((out + "/instances.csv").c_str(), "w");
    std::fprintf(fi, "id,set,n,nvars,nterms,opt,n_trees,enum_min,greedy1,greedy2,greedy,P,z,beta,stages,opt_tree\n");
    FILE* fr = std::fopen((out + "/engine_runs.csv").c_str(), "w");
    std::fprintf(fr, "id,seed,energy,penalty_units,feasible,cost,hit,time_us\n");
    FILE* fc = std::fopen((out + "/energy_check.csv").c_str(), "w");
    std::fprintf(fc, "id,n_states,max_abs_diff\n");
    FILE* fb = std::fopen((out + "/brute_force.csv").c_str(), "w");
    std::fprintf(fb, "id,nvars,states,zero_penalty_states,catalan,all_zero_penalty_are_trees,qubo_min_energy,qubo_min_cost,dp_opt,min_is_feasible,match\n");

    for (auto& in : insts) {
        Model m = build(in);
        std::vector<int> opt_tree; const double opt = dp_opt(m, &opt_tree);
        std::vector<std::vector<int>> trees; enum_trees(m, 0, in.n - 1, trees);
        double emin = 1e300; for (auto& t : trees) emin = std::min(emin, tree_cost(m, t));
        const double g1 = greedy(m, 1, nullptr), g2 = greedy(m, 2, nullptr), g = std::min(g1, g2);
        make_qubo(m, g);
        BinaryProblem p = to_engine(m);
        const int nv = static_cast<int>(m.vars.size());
        // QUBO file for neal
        {
            FILE* fq = std::fopen((out + "/qubo_" + in.id + ".txt").c_str(), "w");
            std::fprintf(fq, "%d %zu %.17g %.17g %.17g\n", nv, m.Qo.size(), m.offset, m.P, opt);
            for (int i = 0; i < nv; ++i) std::fprintf(fq, "%d %.17g %.17g\n", i, m.Qd[i], m.cost[i] / m.P);
            for (auto& [k, v] : m.Qo) std::fprintf(fq, "%d %d %.17g\n", k.first, k.second, v);
            std::fclose(fq);
        }
        // energy check: engine binary_energy vs direct QUBO (minus offset)
        {
            std::mt19937_64 rng(7 + nv); double md = 0;
            for (int t = 0; t < 1000; ++t) {
                std::vector<uint8_t> x(nv); for (auto& b : x) b = rng() & 1u;
                md = std::max(md, std::fabs(binary_energy(p, x) - (qubo_energy(m, x) - m.offset)));
            }
            std::fprintf(fc, "%s,1000,%.3e\n", in.id.c_str(), md);
        }
        // brute force (R1): n=4 first random, n=5 real A and first random
        const bool do_brute = (in.id == "MP_P507-A" || in.id == "R4_0" || in.id == "R5_0");
        if (do_brute) {
            const uint64_t S = 1ull << nv; uint64_t zero = 0; bool all_trees = true; double bestE = 1e300; uint64_t bestx = 0;
            std::vector<uint8_t> x(nv);
            std::vector<long long> hist(400, 0);   // landscape histogram of normalised energy, 0..40 in 0.1
            FILE* fland = in.id == "MP_P507-A" ? std::fopen((out + "/landscape_feasible_MP_P507-A.csv").c_str(), "w") : nullptr;
            if (fland) std::fprintf(fland, "state,energy,cost\n");
            for (uint64_t st = 0; st < S; ++st) {
                for (int i = 0; i < nv; ++i) x[i] = (st >> i) & 1u;
                const double e = qubo_energy(m, x);
                if (e < bestE) { bestE = e; bestx = st; }
                const double pu = penalty_units(m, x);
                if (fland) { const int hb = std::min(399, std::max(0, static_cast<int>(e * 10))); hist[hb]++; }
                if (pu == 0) {
                    ++zero;
                    // is it one of the enumerated trees?
                    bool is_tree = false;
                    for (auto& t : trees) {
                        std::vector<uint8_t> y(nv, 0); for (int v : t) y[v] = 1;
                        if (y == x) { is_tree = true; break; }
                    }
                    all_trees = all_trees && is_tree;
                    if (fland) std::fprintf(fland, "%llu,%.12g,%.12g\n", (unsigned long long)st, e, x_cost(m, x));
                }
            }
            if (fland) {
                std::fclose(fland);
                FILE* fh = std::fopen((out + "/landscape_hist_MP_P507-A.csv").c_str(), "w");
                std::fprintf(fh, "bin_lo,count\n"); for (int i = 0; i < 400; ++i) std::fprintf(fh, "%.1f,%lld\n", i * 0.1, hist[i]);
                std::fclose(fh);
            }
            for (int i = 0; i < nv; ++i) x[i] = (bestx >> i) & 1u;
            const bool feas = penalty_units(m, x) == 0;
            const double bc = x_cost(m, x);
            std::fprintf(fb, "%s,%d,%llu,%llu,%zu,%d,%.12g,%.12g,%.12g,%d,%d\n", in.id.c_str(), nv, (unsigned long long)S,
                         (unsigned long long)zero, trees.size(), all_trees ? 1 : 0, bestE, bc, opt, feas ? 1 : 0,
                         (feas && std::fabs(bc - opt) <= 1e-9 * opt && zero == trees.size() && all_trees) ? 1 : 0);
            std::fflush(fb);
        }
        // engine runs
        for (int sd = 1; sd <= runs; ++sd) {
            const auto t0 = std::chrono::steady_clock::now();
            const std::vector<uint8_t> x = anneal_branch(p, map, static_cast<uint64_t>(sd));
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            const double pu = penalty_units(m, x); const double c = x_cost(m, x);
            const bool hit = pu == 0 && std::fabs(c - opt) <= 1e-9 * opt;
            std::fprintf(fr, "%s,%d,%.12g,%g,%d,%.12g,%d,%.2f\n", in.id.c_str(), sd, qubo_energy(m, x), pu, pu == 0 ? 1 : 0, c, hit ? 1 : 0, us);
        }
        // energy trace (real instances)
        if (in.set != "random") {
            FILE* ft = std::fopen((out + "/energy_trace_" + in.id + ".csv").c_str(), "w");
            std::fprintf(ft, "seed,sweep,energy\n");
            for (int sd = 1; sd <= 8; ++sd) {
                const auto tr = anneal_trace(p, map, sd);
                for (size_t k = 0; k < tr.size(); ++k) std::fprintf(ft, "%d,%zu,%.10g\n", sd, k, tr[k] + m.offset);
            }
            std::fclose(ft);
        }
        std::string zs, bs, ss, ts;
        for (double v : in.z) zs += (zs.empty() ? "" : ";") + std::to_string(v);
        for (double v : in.beta) bs += (bs.empty() ? "" : ";") + std::to_string(v);
        for (int v : m.N) ss += (ss.empty() ? "" : ";") + std::to_string(v);
        for (int v : opt_tree) { const auto& q = m.vars[v]; ts += (ts.empty() ? "" : ";") + std::to_string(q.a) + "-" + std::to_string(q.s) + "-" + std::to_string(q.b); }
        std::fprintf(fi, "%s,%s,%d,%d,%zu,%.12g,%zu,%.12g,%.12g,%.12g,%.12g,%.12g,%s,%s,%s,%s\n", in.id.c_str(), in.set.c_str(), in.n, nv, m.Qo.size(),
                     opt, trees.size(), emin, g1, g2, g, m.P, zs.c_str(), bs.c_str(), ss.c_str(), ts.c_str());
        std::fflush(fi); std::fflush(fr);
        std::printf("%-10s n=%2d vars=%3d opt=%.4f greedy=%.4f trees=%zu\n", in.id.c_str(), in.n, nv, opt, g, trees.size());
        std::fflush(stdout);
    }
    std::fclose(fi); std::fclose(fr); std::fclose(fc); std::fclose(fb);
    return 0;
}
