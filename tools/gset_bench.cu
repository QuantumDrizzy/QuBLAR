// Gset MaxCut with the engine's own annealer. ADR-008: the QUBO is the interface.
//
// Schedule copied from check_ising's oracle (MAP), before any Gset number was
// read: t_hot = 5, t_cold = 1e-3, 400 anneal sweeps, 20 hold sweeps, 8 seeds.
// lambda = -1 so a disagreeing edge lowers the energy and the cut is -E.
// Edges must be weight +1. A negative weight is a different QUBO this binary
// does not pretend to solve.
//
// ADR-015 (P1) adds three modes; the legacy mode below is unchanged, byte for byte
// in its output, and still runs anneal_branch:
//   gset_bench <graph>                          legacy: frozen schedule, seeds 1..8
//   gset_bench --check [gset_dir]               fast self-check for check.bat
//   gset_bench --repro <gset_dir> <out.csv>     ADR-015 R1(b): integer engine vs anneal_branch, G1-G5
//   gset_bench --suite <list> <outdir> [threads=16] [cap_s=1800]
//                                               ADR-015 cells: B1/B10/B100, runs.csv + summary.csv
// The integer path (per-edge weights, exact int64 energy) is src/ising_weighted.hpp.

#include "ising_recon.hpp"
#include "ising_weighted.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace argos;

static const double kHot = 5.0;
static const double kCold = 1e-3;
static const int kAnneal = 400;
static const int kHold = 20;
static const int kSeeds = 8;

static int cut_of(const std::vector<std::pair<int, int>>& edges, const std::vector<uint8_t>& x) {
    int cut = 0;
    for (const auto& e : edges)
        cut += x[e.first] != x[e.second];
    return cut;
}

static BinaryProblem problem_of(int n, const std::vector<std::pair<int, int>>& edges) {
    BinaryProblem p;
    p.var_voxel.resize(n);
    p.voxel_var.resize(n);
    for (int i = 0; i < n; ++i) {
        p.var_voxel[i] = i;
        p.voxel_var[i] = i;
    }
    p.lambda = -1.0;
    p.kappa = 0.0;
    p.row_ptr.assign(n + 1, 0);
    std::vector<std::vector<int>> adj(n);
    for (const auto& e : edges) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
    }
    p.nbr_ptr.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        for (int u : adj[i]) p.nbr.push_back(u);
        p.nbr_ptr[i + 1] = int(p.nbr.size());
    }
    return p;
}

static bool one_edge_is_cut() {
    std::vector<std::pair<int, int>> edges = {{0, 1}};
    BinaryProblem p = problem_of(2, edges);
    Schedule map;
    map.t_hot = kHot;
    map.t_cold = kCold;
    map.anneal_sweeps = kAnneal;
    map.hold_sweeps = kHold;
    const std::vector<uint8_t> x = anneal_branch(p, map, 1);
    const int cut = cut_of(edges, x);
    const double energy = binary_energy(p, x);
    return cut == 1 && energy == -1.0;
}

static int legacy_main(int argc, char** argv) {
    if (!one_edge_is_cut()) {
        std::fprintf(stderr, "oracle: a single edge was not cut\n");
        return 2;
    }
    if (argc < 2) {
        std::fprintf(stderr, "usage: gset_bench <graph>\n");
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    int n = 0, m = 0;
    in >> n >> m;
    std::vector<std::pair<int, int>> edges;
    edges.reserve(m);
    for (int e = 0; e < m; ++e) {
        int u = 0, v = 0, w = 1;
        in >> u >> v;
        if (in.peek() != '\n' && in.peek() != EOF) in >> w;
        if (w != 1) {
            std::fprintf(stderr, "edge weight %d is not +1; this bench does not solve it\n", w);
            return 2;
        }
        edges.push_back({u - 1, v - 1});
    }
    BinaryProblem p = problem_of(n, edges);
    Schedule map;
    map.t_hot = kHot;
    map.t_cold = kCold;
    map.anneal_sweeps = kAnneal;
    map.hold_sweeps = kHold;

    int best_cut = -1;
    int best_seed = 0;
    double best_energy = 0.0;
    std::vector<uint8_t> best;
    for (int seed = 1; seed <= kSeeds; ++seed) {
        const std::vector<uint8_t> x = anneal_branch(p, map, uint64_t(seed));
        const int cut = cut_of(edges, x);
        const double energy = binary_energy(p, x);
        if (energy != -double(cut)) {
            std::fprintf(stderr, "energy %.1f is not -cut %d\n", energy, cut);
            return 2;
        }
        if (cut > best_cut) {
            best_cut = cut;
            best_seed = seed;
            best_energy = energy;
            best = x;
        }
    }
    std::printf("graph %s\n", argv[1]);
    std::printf("n %d\n", n);
    std::printf("m %d\n", int(edges.size()));
    std::printf("schedule map t_hot %.1f t_cold %.0e anneal %d hold %d seeds %d\n",
                kHot, kCold, kAnneal, kHold, kSeeds);
    std::printf("cut %d\n", best_cut);
    std::printf("energy %.1f\n", best_energy);
    std::printf("seed %d\n", best_seed);
    std::printf("assignment ");
    for (uint8_t b : best) std::printf("%u", unsigned(b));
    std::printf("\n");
    return 0;
}

// =============================================================================
// ADR-015 modes
// =============================================================================

namespace p1 {

using argos::wising::WeightedGraph;
using argos::wising::Budget;

static int g_fail = 0;

static void line(const char* what, bool ok, const std::string& detail) {
    std::printf("  %-52s %s  %s\n", what, ok ? "PASS" : "FAIL", detail.c_str());
    if (!ok) ++g_fail;
}

static bool parse_text(const std::string& text, WeightedGraph& g, std::string& err) {
    return argos::wising::parse_rudy(text.data(), text.size(), g, err);
}

static bool load_graph(const std::string& path, WeightedGraph& g, std::string& err,
                       std::string* sha = nullptr) {
    std::vector<char> buf;
    if (!argos::wising::read_file(path, buf)) { err = "cannot read " + path; return false; }
    if (sha) *sha = argos::wising::Sha256::of(buf.data(), buf.size());
    return argos::wising::parse_rudy(buf.data(), buf.size(), g, err);
}

static std::vector<std::pair<int, int>> edges_of(const WeightedGraph& g) {
    std::vector<std::pair<int, int>> e(static_cast<size_t>(g.m));
    for (int64_t k = 0; k < g.m; ++k) e[size_t(k)] = {g.eu[size_t(k)], g.ev[size_t(k)]};
    return e;
}

static Schedule legacy_schedule() {
    Schedule s;
    s.t_hot = kHot;
    s.t_cold = kCold;
    s.anneal_sweeps = kAnneal;
    s.hold_sweeps = kHold;
    return s;
}

static Budget budget_of(int anneal, int hold) {
    Budget b;
    b.t_hot = kHot;
    b.t_cold = kCold;
    b.anneal_sweeps = anneal;
    b.hold_sweeps = hold;
    return b;
}

static std::string fmt(double v) {
    if (std::isinf(v)) return "inf";
    char b[64];
    std::snprintf(b, sizeof b, "%.6f", v);
    return b;
}

// Best assignments of RESULTS-gset.md (frozen reference for ADR-015 R1(b)).
static const char* kGsetSha[5] = {
    "6fa9d0c7e36c590d3d8622e6f548489db688194981ca1f1c8f759b0dd1500a98",
    "f59384c0a7495284c74e70c5569c57071db06024322df7f1d456962f9730beab",
    "20616526f28f11564a20e13cf7a71d6039c3b5ecd64213a520b3dd107a01b567",
    "55156760958f81d0f2f1b5a1d601363f03e38cb38c5e36ca362d7800f5fe5649",
    "936ddbc2c0167d4bf22d02147aef02c236072bddb60018aeaf24c2f4149ca5a7"};
static const int kGsetCut[5] = {11613, 11617, 11621, 11646, 11616};

struct Repro { int same = 0, total = 0, best_cut = -1; std::string best_sha; bool consistent = true; };

// Integer engine vs anneal_branch on a +1 graph, seeds 1..seeds, frozen schedule.
static Repro compare_engines(const WeightedGraph& g, int seeds, std::FILE* csv, const char* name) {
    Repro r;
    const auto edges = edges_of(g);
    const BinaryProblem p = problem_of(g.n, edges);
    const Schedule map = legacy_schedule();
    const Budget b = budget_of(kAnneal, kHold);
    for (int seed = 1; seed <= seeds; ++seed) {
        const std::vector<uint8_t> xl = anneal_branch(p, map, uint64_t(seed));
        const argos::wising::RunOut w = argos::wising::anneal(g, b, uint64_t(seed));
        const bool same = xl == w.x;
        const int64_t cut = argos::wising::cut_of(g, w.x);
        const int64_t es = argos::wising::energy_scratch(g, w.x);
        const bool cons = cut == -w.energy_tracked && cut == -es && double(cut) == -binary_energy(p, xl);
        r.consistent = r.consistent && cons;
        r.same += same;
        ++r.total;
        if (int(cut) > r.best_cut) { r.best_cut = int(cut); r.best_sha = argos::wising::assignment_sha256(w.x); }
        if (csv)
            std::fprintf(csv, "%s,%d,%d,%lld,%lld,%lld,%d,%s\n", name, seed, int(same), (long long)cut,
                         (long long)w.energy_tracked, (long long)es, int(cons),
                         argos::wising::assignment_sha256(w.x).c_str());
    }
    return r;
}

static uint64_t mix(uint64_t& s) { return argos::wising::detail::splitmix(s); }

static WeightedGraph random_complete(int n, int wmax, uint64_t seed) {
    std::string t = std::to_string(n) + " " + std::to_string(n * (n - 1) / 2) + "\n";
    uint64_t s = seed;
    for (int i = 1; i <= n; ++i)
        for (int j = i + 1; j <= n; ++j) {
            const int w = int(mix(s) % uint64_t(2 * wmax + 1)) - wmax;
            t += std::to_string(i) + " " + std::to_string(j) + " " + std::to_string(w) + "\n";
        }
    WeightedGraph g;
    std::string err;
    parse_text(t, g, err);
    return g;
}

static WeightedGraph random_sparse_plus1(int n, int m, uint64_t seed) {
    uint64_t s = seed;
    std::string t = std::to_string(n) + " " + std::to_string(m) + "\n";
    std::vector<char> seen(size_t(n) * n, 0);
    int made = 0;
    while (made < m) {
        const int u = int(mix(s) % uint64_t(n)), v = int(mix(s) % uint64_t(n));
        if (u == v) continue;
        const int a = std::min(u, v), b = std::max(u, v);
        if (seen[size_t(a) * n + b]) continue;
        seen[size_t(a) * n + b] = 1;
        t += std::to_string(a + 1) + " " + std::to_string(b + 1) + "\n";
        ++made;
    }
    WeightedGraph g;
    std::string err;
    parse_text(t, g, err);
    return g;
}

// Gray-code walk over all 2^n states: tracked energy and predicted dE against
// the edge-list cut and the CSR scratch energy, at every state.
static int64_t gray_check(const WeightedGraph& g, int64_t* best_cut) {
    argos::wising::State st(g);
    std::vector<uint8_t> x(size_t(g.n), 0);
    int64_t bad = 0;
    int64_t best = 0;
    const uint64_t states = uint64_t(1) << g.n;
    for (uint64_t k = 1; k < states; ++k) {
        int bit = 0;
        while (!((k >> bit) & 1)) ++bit;
        const int64_t before = st.energy;
        const int64_t predicted = st.dE(bit);
        st.flip(bit);
        x[size_t(bit)] ^= 1u;
        const int64_t cut = argos::wising::cut_of(g, x);
        if (st.energy != -cut || st.energy != argos::wising::energy_scratch(g, x) ||
            st.energy - before != predicted)
            ++bad;
        best = std::max(best, cut);
    }
    if (best_cut) *best_cut = best;
    return bad;
}

static int mode_check(int argc, char** argv) {
    std::printf("QuBLAR -- gset_bench self-check (ADR-015)\n");
    // A. the loader refuses what it cannot represent exactly
    struct Bad { const char* text; const char* why; };
    const Bad bad[] = {{"3 1\n1 2 0.5\n", "non-integer weight"},
                       {"3 1\n1 1 1\n", "self-loop"},
                       {"3 2\n1 2 1\n2 1 1\n", "duplicate edge"},
                       {"3 2\n1 2 1\n", "edge count"},
                       {"3 1\n1 4 1\n", "index out of range"}};
    int refused = 0;
    for (const Bad& b : bad) {
        WeightedGraph g;
        std::string err;
        refused += parse_text(b.text, g, err) ? 0 : 1;
    }
    line("the loader refuses 5 malformed inputs", refused == 5,
         std::to_string(refused) + " of 5 refused");
    {
        WeightedGraph g;
        std::string err;
        const bool ok = parse_text("3 2\n1 2\n2 3 -7\n", g, err);
        line("a missing weight is +1, a signed weight is kept", ok && g.sum_w == -6 && g.m == 2,
             ok ? "sum of weights " + std::to_string(g.sum_w) : err);
    }
    // B. exact energies over every state of three integer-weighted K16
    int64_t mism = 0;
    std::string opt_detail;
    bool near = true;
    for (uint64_t seed = 1; seed <= 3; ++seed) {
        const WeightedGraph g = random_complete(16, 1000, seed);
        int64_t best = 0;
        mism += gray_check(g, &best);
        int64_t eng = std::numeric_limits<int64_t>::min();
        for (int s = 1; s <= 8; ++s) {
            const auto r = argos::wising::anneal(g, budget_of(4000, 200), uint64_t(s));
            eng = std::max(eng, argos::wising::cut_of(g, r.x));
        }
        near = near && double(eng) >= 0.99 * double(best);
        opt_detail += (seed > 1 ? ", " : "") + std::to_string(eng) + "/" + std::to_string(best);
    }
    line("integer energy exact on all 2^16 states, 3 K16", mism == 0,
         std::to_string(mism) + " mismatches over 196605 flips");
    std::printf("    engine best of 8 (B10) / exact optimum: %s (reported, not decisive: %s)\n",
                opt_detail.c_str(), near ? ">= 0.99" : "< 0.99 somewhere");
    // C. the integer engine makes anneal_branch's decisions on a +1 graph
    {
        const WeightedGraph g = random_sparse_plus1(300, 3000, 42);
        const Repro r = compare_engines(g, 4, nullptr, "synthetic");
        line("integer engine = anneal_branch, synthetic +1 graph", r.same == r.total && r.consistent,
             std::to_string(r.same) + " of " + std::to_string(r.total) + " seeds identical");
    }
    // D. on G1, if the data are present
    const std::string dir = argc > 2 ? argv[2] : "data\\gset";
    WeightedGraph g1;
    std::string err;
    if (load_graph(dir + "\\G1", g1, err)) {
        const Repro r = compare_engines(g1, 8, nullptr, "G1");
        line("G1: integer engine = anneal_branch, 8 seeds", r.same == 8 && r.consistent,
             std::to_string(r.same) + " of 8 identical");
        line("G1: best cut and assignment hash as committed",
             r.best_cut == kGsetCut[0] && r.best_sha == kGsetSha[0],
             "cut " + std::to_string(r.best_cut));
    } else {
        std::printf("  %-52s SKIP  (no data: %s)\n", "G1 against the committed Gset numbers", err.c_str());
    }
    std::printf(g_fail ? "\nFAILED: %d\n" : "\nall checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}

static int mode_repro(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: gset_bench --repro <gset_dir> <out.csv>\n"); return 2; }
    std::FILE* csv = std::fopen(argv[3], "wb");
    if (!csv) { std::fprintf(stderr, "cannot write %s\n", argv[3]); return 2; }
    std::fprintf(csv, "instance,seed,identical_to_anneal_branch,cut,energy_tracked,energy_scratch,consistent,assignment_sha256\n");
    std::printf("ADR-015 R1(b): integer engine vs anneal_branch, frozen schedule, seeds 1..8\n");
    int same = 0, total = 0;
    bool ok_all = true;
    for (int k = 1; k <= 5; ++k) {
        const std::string name = "G" + std::to_string(k);
        WeightedGraph g;
        std::string err;
        if (!load_graph(std::string(argv[2]) + "\\" + name, g, err)) {
            std::printf("  %s: %s\n", name.c_str(), err.c_str());
            ok_all = false;
            continue;
        }
        const Repro r = compare_engines(g, 8, csv, name.c_str());
        same += r.same;
        total += r.total;
        const bool ok = r.same == 8 && r.consistent && r.best_cut == kGsetCut[k - 1] &&
                        r.best_sha == kGsetSha[k - 1];
        ok_all = ok_all && ok;
        std::printf("  %s: %d/8 identical, best cut %d (committed %d), sha %s, consistent %s -> %s\n",
                    name.c_str(), r.same, r.best_cut, kGsetCut[k - 1],
                    r.best_sha == kGsetSha[k - 1] ? "matches" : "DIFFERS",
                    r.consistent ? "yes" : "NO", ok ? "ok" : "MISMATCH");
    }
    std::fclose(csv);
    std::printf("R1(b): %d of %d assignments identical -> %s\n", same, total,
                ok_all && same == 40 ? "PASS" : "FAIL");
    return ok_all && same == 40 ? 0 : 1;
}

struct Cell { const char* name; int anneal, hold, runs; };
static const Cell kCells[3] = {{"B1", 400, 20, 64}, {"B10", 4000, 200, 64}, {"B100", 40000, 2000, 16}};

static int mode_suite(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: gset_bench --suite <list> <outdir> [threads=16] [cap_s=1800]\n");
        return 2;
    }
    const std::string outdir = argv[3];
    const int threads = argc > 4 ? std::atoi(argv[4]) : 16;
    const double cap = argc > 5 ? std::atof(argv[5]) : 1800.0;
    std::FILE* list = std::fopen(argv[2], "rb");
    if (!list) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 2; }
    std::FILE* runs = std::fopen((outdir + "\\runs.csv").c_str(), "wb");
    std::FILE* summ = std::fopen((outdir + "\\summary.csv").c_str(), "wb");
    std::FILE* load = std::fopen((outdir + "\\load.csv").c_str(), "wb");
    if (!runs || !summ || !load) { std::fprintf(stderr, "cannot write into %s\n", outdir.c_str()); return 2; }
    std::fprintf(runs, "instance,budget,anneal_sweeps,hold_sweeps,seed,cut,energy_tracked,energy_scratch,consistent,success,wall_s,assignment_sha256\n");
    std::fprintf(summ, "instance,n,m,budget,anneal_sweeps,hold_sweeps,runs_planned,runs_done,best_cut,target,gap_pct,mean_cut,p_success,t_mean_s,tts99_s,tts99_ci_lo_s,tts99_ci_hi_s,all_consistent,best_seed,best_sha256\n");
    std::fprintf(load, "instance,path,sha256,sha256_expected,sha_ok,n,m,n_plus1,n_minus1,n_other,sum_w,expect_ok,loaded,error\n");
    char buf[4096];
    int cells_done = 0, cells_total = 0, loaded = 0, listed = 0;
    int64_t inconsistent = 0;
    while (std::fgets(buf, sizeof buf, list)) {
        char name[64], path[1024], sha_exp[80];
        long long target = 0, en = -1, em = -1, ep = -1, eneg = -1;
        const int got = std::sscanf(buf, "%63s %1023s %lld %79s %lld %lld %lld %lld", name, path, &target,
                                    sha_exp, &en, &em, &ep, &eneg);
        if (got < 4 || name[0] == '#') continue;
        ++listed;
        WeightedGraph g;
        std::string err, sha;
        const bool ok_load = load_graph(path, g, err, &sha);
        const bool sha_ok = sha == sha_exp;
        bool expect_ok = ok_load;
        if (ok_load && got >= 6) expect_ok = expect_ok && g.n == en && g.m == em;
        if (ok_load && got >= 8) expect_ok = expect_ok && g.n_pos == ep && g.n_neg == eneg;
        std::fprintf(load, "%s,%s,%s,%s,%d,%d,%lld,%lld,%lld,%lld,%lld,%d,%d,%s\n", name, path, sha.c_str(),
                     sha_exp, int(sha_ok), g.n, (long long)g.m, (long long)g.n_pos, (long long)g.n_neg,
                     (long long)g.n_other, (long long)g.sum_w, int(expect_ok), int(ok_load), err.c_str());
        std::fflush(load);
        std::printf("%s: n %d m %lld, sha %s, %s\n", name, g.n, (long long)g.m, sha_ok ? "ok" : "MISMATCH",
                    ok_load ? (expect_ok ? "loaded" : "loaded, counts DIFFER") : err.c_str());
        if (!ok_load) { cells_total += 3; continue; }
        if (sha_ok && expect_ok) ++loaded;
        for (const Cell& c : kCells) {
            ++cells_total;
            const Budget b = budget_of(c.anneal, c.hold);
            std::vector<int64_t> cut(size_t(c.runs), 0), etr(size_t(c.runs), 0), esc(size_t(c.runs), 0);
            std::vector<double> wall(size_t(c.runs), 0.0);
            std::vector<std::string> shas(static_cast<size_t>(c.runs));
            std::vector<char> done(size_t(c.runs), 0);
            std::atomic<int> next{0};
            const auto t0 = std::chrono::steady_clock::now();
            auto worker = [&]() {
                for (;;) {
                    const int k = next++;
                    if (k >= c.runs) return;
                    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    if (el > cap) continue;
                    const auto a = std::chrono::steady_clock::now();
                    const argos::wising::RunOut r = argos::wising::anneal(g, b, uint64_t(k + 1));
                    wall[size_t(k)] = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
                    cut[size_t(k)] = argos::wising::cut_of(g, r.x);
                    etr[size_t(k)] = r.energy_tracked;
                    esc[size_t(k)] = argos::wising::energy_scratch(g, r.x);
                    shas[size_t(k)] = argos::wising::assignment_sha256(r.x);
                    done[size_t(k)] = 1;
                }
            };
            std::vector<std::thread> pool;
            for (int t = 0; t < std::max(1, threads); ++t) pool.emplace_back(worker);
            for (auto& th : pool) th.join();
            const double cell_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            int n_done = 0, succ_n = 0, best_seed = 0;
            int64_t best = std::numeric_limits<int64_t>::min();
            double sum_cut = 0.0, sum_t = 0.0;
            bool all_cons = true;
            std::vector<int> succ;
            std::vector<double> times;
            for (int k = 0; k < c.runs; ++k) {
                if (!done[size_t(k)]) continue;
                ++n_done;
                const bool cons = cut[size_t(k)] == -etr[size_t(k)] && cut[size_t(k)] == -esc[size_t(k)];
                all_cons = all_cons && cons;
                inconsistent += cons ? 0 : 1;
                const int s = cut[size_t(k)] >= target ? 1 : 0;
                succ_n += s;
                succ.push_back(s);
                times.push_back(wall[size_t(k)]);
                sum_cut += double(cut[size_t(k)]);
                sum_t += wall[size_t(k)];
                if (cut[size_t(k)] > best) { best = cut[size_t(k)]; best_seed = k + 1; }
                std::fprintf(runs, "%s,%s,%d,%d,%d,%lld,%lld,%lld,%d,%d,%s,%s\n", name, c.name, c.anneal, c.hold,
                             k + 1, (long long)cut[size_t(k)], (long long)etr[size_t(k)], (long long)esc[size_t(k)],
                             int(cons), s, fmt(wall[size_t(k)]).c_str(), shas[size_t(k)].c_str());
            }
            std::fflush(runs);
            if (n_done == c.runs) ++cells_done;
            const double p = n_done ? double(succ_n) / n_done : 0.0;
            const double tm = n_done ? sum_t / n_done : 0.0;
            const double tts = n_done ? argos::wising::tts99(tm, p) : std::numeric_limits<double>::infinity();
            argos::wising::TtsCi ci{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
            if (n_done) ci = argos::wising::tts99_bootstrap(succ, times);
            const double gap = n_done ? 100.0 * double(best - target) / double(target) : 0.0;
            std::fprintf(summ, "%s,%d,%lld,%s,%d,%d,%d,%d,%lld,%lld,%s,%s,%s,%s,%s,%s,%s,%d,%d,%s\n", name, g.n,
                         (long long)g.m, c.name, c.anneal, c.hold, c.runs, n_done, (long long)best, target,
                         fmt(gap).c_str(), fmt(n_done ? sum_cut / n_done : 0.0).c_str(), fmt(p).c_str(),
                         fmt(tm).c_str(), fmt(tts).c_str(), fmt(ci.lo).c_str(), fmt(ci.hi).c_str(), int(all_cons),
                         best_seed, best_seed ? shas[size_t(best_seed - 1)].c_str() : "");
            std::fflush(summ);
            std::printf("  %-5s %-4s runs %d/%d  best %lld (target %lld, gap %s%%)  p %s  t %s s  TTS99 %s s [%s, %s]  consistent %s  (%.1f s)\n",
                        name, c.name, n_done, c.runs, (long long)best, target, fmt(gap).c_str(), fmt(p).c_str(),
                        fmt(tm).c_str(), fmt(tts).c_str(), fmt(ci.lo).c_str(), fmt(ci.hi).c_str(),
                        all_cons ? "yes" : "NO", cell_s);
            std::fflush(stdout);
        }
    }
    std::fclose(list);
    std::fclose(runs);
    std::fclose(summ);
    std::fclose(load);
    std::printf("instances listed %d, loaded with matching hash and counts %d; cells complete %d of %d; inconsistent runs %lld\n",
                listed, loaded, cells_done, cells_total, (long long)inconsistent);
    return 0;
}

}  // namespace p1

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--check") == 0) return p1::mode_check(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "--repro") == 0) return p1::mode_repro(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "--suite") == 0) return p1::mode_suite(argc, argv);
    return legacy_main(argc, argv);
}
