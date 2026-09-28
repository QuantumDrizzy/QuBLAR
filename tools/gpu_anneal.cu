// =============================================================================
// QuBLAR -- gpu_anneal: the ADR-016 GPU engine against the P1 CPU engine
// =============================================================================
//   gpu_anneal --check [gset_dir]      fast self-check (check.bat)
//   gpu_anneal --bench <list> <outdir> the ADR-016 section 4 suite
// The CPU side is argos::wising::anneal, unmodified (src/ising_weighted.hpp).
// =============================================================================

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ising_gpu.cuh"

using argos::wising::Budget;
using argos::wising::WeightedGraph;
namespace wi = argos::wising;
namespace gi = argos::gising;

static int g_fail = 0;

static void line(const char* what, bool ok, const std::string& detail) {
    std::printf("  %-60s %s  %s\n", what, ok ? "PASS" : "FAIL", detail.c_str());
    std::fflush(stdout);
    if (!ok) ++g_fail;
}

static std::string fmt(double v) {
    if (std::isinf(v)) return "inf";
    char b[64];
    std::snprintf(b, sizeof b, "%.6f", v);
    return b;
}

static uint64_t mix(uint64_t& s) { return wi::detail::splitmix(s); }

static bool load_graph(const std::string& path, WeightedGraph& g, std::string& err, std::string* sha = nullptr) {
    std::vector<char> buf;
    if (!wi::read_file(path, buf)) { err = "cannot read " + path; return false; }
    if (sha) *sha = wi::Sha256::of(buf.data(), buf.size());
    return wi::parse_rudy(buf.data(), buf.size(), g, err);
}

// Same generators as tools/gset_bench.cu (ADR-015 self-check).
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
    wi::parse_rudy(t.data(), t.size(), g, err);
    return g;
}

static WeightedGraph random_sparse(int n, int m, uint64_t seed, int wmax) {
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
        t += std::to_string(a + 1) + " " + std::to_string(b + 1);
        if (wmax > 0) {
            int w = 0;
            while (w == 0) w = int(mix(s) % uint64_t(2 * wmax + 1)) - wmax;
            t += " " + std::to_string(w);
        }
        t += "\n";
        ++made;
    }
    WeightedGraph g;
    std::string err;
    wi::parse_rudy(t.data(), t.size(), g, err);
    return g;
}

static std::string plan_str(const gi::Plan& p) {
    char b[160];
    std::snprintf(b, sizeof b, "%s, w int%d, f int%d, %s state", p.dense ? "dense" : "CSR", 8 * p.weight_bytes,
                  8 * p.field_bytes, p.shared_state ? "shared" : "global");
    return b;
}

template <typename Fn>
static void parallel_for(int count, int threads, Fn fn) {
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&]() {
            for (;;) {
                const int k = next++;
                if (k >= count) return;
                fn(k);
            }
        });
    for (auto& th : pool) th.join();
}

// Random 0/1 states: splitmix64 seeded `seed`, bit = top bit of each draw, spins in order.
static std::vector<uint8_t> random_states(int K, int n, uint64_t seed) {
    std::vector<uint8_t> X(size_t(K) * size_t(n));
    uint64_t s = seed;
    for (size_t k = 0; k < X.size(); ++k) X[k] = uint8_t(mix(s) >> 63);
    return X;
}

static int64_t oracle_mismatches(const WeightedGraph& g, const gi::DeviceGraph& dg, int K, uint64_t seed) {
    const std::vector<uint8_t> X = random_states(K, g.n, seed);
    const std::vector<long long> ed = gi::device_energies(dg, X, K);
    int64_t bad = 0;
    for (int k = 0; k < K; ++k) {
        const std::vector<uint8_t> x(X.begin() + size_t(k) * g.n, X.begin() + size_t(k + 1) * g.n);
        if (ed[size_t(k)] != -wi::cut_of(g, x)) ++bad;
    }
    return bad;
}

struct Compare { int same = 0, total = 0, consistent = 0; };

// GPU batch (seeds 1..R) against the CPU engine for the same seeds.
static Compare compare_gpu_cpu(const WeightedGraph& g, bool force_global, int anneal, int hold, int R) {
    const gi::Plan p = gi::make_plan(g, force_global);
    const gi::DeviceGraph dg(g, p);
    const Budget b = gi::scaled_budget(p, anneal, hold);
    const gi::BatchOut out = gi::run_batch(dg, b, 1, R);
    Compare c;
    std::vector<int> same(size_t(R), 0), cons(size_t(R), 0);
    parallel_for(R, 16, [&](int r) {
        const wi::RunOut cpu = wi::anneal(g, b, uint64_t(r + 1));
        const std::vector<uint8_t> x = out.x(r);
        const int64_t cut = wi::cut_of(g, x);
        same[size_t(r)] = x == cpu.x && out.energy[size_t(r)] == cpu.energy_tracked;
        cons[size_t(r)] = cut == -out.energy[size_t(r)] && cut == -wi::energy_scratch(g, x);
    });
    for (int r = 0; r < R; ++r) { c.same += same[size_t(r)]; c.consistent += cons[size_t(r)]; ++c.total; }
    return c;
}

static int mode_check(int argc, char** argv) {
    std::printf("QuBLAR -- gpu_anneal self-check (ADR-016)\n");
    {
        int64_t bad = 0;
        for (uint64_t seed = 1; seed <= 3; ++seed) {
            const WeightedGraph g = random_complete(16, 1000, seed);
            const gi::DeviceGraph dg(g, gi::make_plan(g));
            bad += oracle_mismatches(g, dg, 1000, 20260929ull + seed);
        }
        line("device energy = host energy, 3 K16 x 1000 random states", bad == 0,
             std::to_string(bad) + " mismatches of 3000");
    }
    {
        const WeightedGraph g = random_complete(16, 1000, 1);
        const Compare c = compare_gpu_cpu(g, false, 400, 20, 8);
        line(("GPU = CPU engine, K16 (" + plan_str(gi::make_plan(g)) + ")").c_str(),
             c.same == 8 && c.consistent == 8, std::to_string(c.same) + " of 8 seeds identical");
    }
    const WeightedGraph sp = random_sparse(300, 3000, 42, 0);
    {
        const Compare c = compare_gpu_cpu(sp, false, 400, 20, 8);
        line(("GPU = CPU engine, sparse +1 (" + plan_str(gi::make_plan(sp)) + ")").c_str(),
             c.same == 8 && c.consistent == 8, std::to_string(c.same) + " of 8 seeds identical");
    }
    {
        const Compare c = compare_gpu_cpu(sp, true, 400, 20, 8);
        line(("GPU = CPU engine, sparse +1 (" + plan_str(gi::make_plan(sp, true)) + ")").c_str(),
             c.same == 8 && c.consistent == 8, std::to_string(c.same) + " of 8 seeds identical");
    }
    {
        const WeightedGraph sw = random_sparse(300, 3000, 7, 1000);
        const Compare c = compare_gpu_cpu(sw, false, 400, 20, 8);
        line(("GPU = CPU engine, signed sparse (" + plan_str(gi::make_plan(sw)) + ")").c_str(),
             c.same == 8 && c.consistent == 8, std::to_string(c.same) + " of 8 seeds identical");
    }
    const std::string dir = argc > 2 ? argv[2] : "data\\gset";
    WeightedGraph g1;
    std::string err;
    if (load_graph(dir + "\\G1", g1, err)) {
        const Compare c = compare_gpu_cpu(g1, false, 400, 20, 64);
        line("G1: GPU = CPU engine, 64 seeds, scaled T, 400+20", c.same == 64 && c.consistent == 64,
             std::to_string(c.same) + " of 64 identical, " + std::to_string(c.consistent) + " consistent");
    } else {
        std::printf("  %-60s SKIP  (no data: %s)\n", "G1: GPU = CPU engine", err.c_str());
    }
    std::printf(g_fail ? "\nFAILED: %d\n" : "\nall checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}

// ---- the ADR-016 suite -------------------------------------------------------------

struct Stat {
    int runs = 0;
    double wall = 0.0;
    int64_t best = std::numeric_limits<int64_t>::min();
    int best_seed = 0;
    double sum_cut = 0.0;
    int succ = 0;
    int64_t inconsistent = 0;
    std::vector<int> success;
};

static int mode_bench(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: gpu_anneal --bench <list> <outdir> [R=16384] [threads=16]\n"); return 2; }
    const std::string outdir = argv[3];
    const int R = argc > 4 ? std::atoi(argv[4]) : 16384;
    const int threads = argc > 5 ? std::atoi(argv[5]) : 16;
    const int A = 4000, H = 200;
    std::FILE* list = std::fopen(argv[2], "rb");
    if (!list) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 2; }
    auto open = [&](const char* name) { return std::fopen((outdir + "\\" + name).c_str(), "wb"); };
    std::FILE* fg = open("runs_gpu.csv");
    std::FILE* fc = open("runs_cpu.csv");
    std::FILE* fs = open("summary.csv");
    std::FILE* ft = open("throughput.csv");
    std::FILE* fa = open("agreement.csv");
    std::FILE* fl = open("load.csv");
    if (!fg || !fc || !fs || !ft || !fa || !fl) { std::fprintf(stderr, "cannot write into %s\n", outdir.c_str()); return 2; }
    std::fprintf(fg, "instance,seed,cut,energy_tracked,energy_scratch,consistent,success,identical_to_cpu,assignment_sha256\n");
    std::fprintf(fc, "instance,seed,cut,energy_tracked,energy_scratch,consistent,success,wall_s,assignment_sha256\n");
    std::fprintf(fs, "instance,device,n,m,t_hot,t_cold,anneal_sweeps,hold_sweeps,runs,wall_s,spin_updates,rate_per_s,best_cut,target,gap_pct,mean_cut,p_success,t_amortized_s,tts99_s,tts99_ci_lo_s,tts99_ci_hi_s,latency_s,tts99_latency_s,all_consistent,best_seed\n");
    std::fprintf(ft, "instance,plan,rate_cpu_per_s,rate_gpu_per_s,speedup,wall_gpu_s,wall_cpu_s,kernel_ms,launches,sweeps_per_launch,runs_gpu,runs_cpu\n");
    std::fprintf(fa, "instance,shared_seeds,identical,oracle_states,oracle_mismatches\n");
    std::fprintf(fl, "instance,path,sha256,sha256_expected,sha_ok,n,m,n_plus1,n_minus1,counts_ok,sigma,w_min,field_bound,plan\n");
    char buf[4096];
    int idx = 0;
    while (std::fgets(buf, sizeof buf, list)) {
        char name[64], path[1024], sha_exp[80];
        long long target = 0, en = -1, em = -1, ep = -1, eneg = -1;
        const int got = std::sscanf(buf, "%63s %1023s %lld %79s %lld %lld %lld %lld", name, path, &target, sha_exp,
                                    &en, &em, &ep, &eneg);
        if (got < 4 || name[0] == '#') continue;
        ++idx;
        WeightedGraph g;
        std::string err, sha;
        if (!load_graph(path, g, err, &sha)) { std::printf("%s: %s\n", name, err.c_str()); continue; }
        const bool sha_ok = sha == sha_exp;
        const bool counts_ok = (got < 6 || (g.n == en && g.m == em)) && (got < 8 || (g.n_pos == ep && g.n_neg == eneg));
        const gi::Plan p = gi::make_plan(g);
        std::fprintf(fl, "%s,%s,%s,%s,%d,%d,%lld,%lld,%lld,%d,%s,%lld,%lld,%s\n", name, path, sha.c_str(), sha_exp,
                     int(sha_ok), g.n, (long long)g.m, (long long)g.n_pos, (long long)g.n_neg, int(counts_ok),
                     fmt(p.sigma).c_str(), (long long)p.w_min, (long long)p.B, plan_str(p).c_str());
        std::fflush(fl);
        const Budget b = gi::scaled_budget(p, A, H);
        std::printf("%s: n %d m %lld, sha %s, counts %s, %s, T_hot %.3f T_cold %.3f\n", name, g.n, (long long)g.m,
                    sha_ok ? "ok" : "MISMATCH", counts_ok ? "ok" : "DIFFER", plan_str(p).c_str(), b.t_hot, b.t_cold);
        std::fflush(stdout);
        if (!p.refused.empty()) { std::printf("  refused: %s\n", p.refused.c_str()); continue; }
        const gi::DeviceGraph dg(g, p);
        const int64_t orc_bad = oracle_mismatches(g, dg, 1000, 20260929ull + uint64_t(idx));
        // GPU batch
        const gi::BatchOut out = gi::run_batch(dg, b, 1, R);
        std::printf("  GPU: %d replicas, wall %.3f s (kernels %.1f ms, %d launches of %d sweeps)\n", R, out.wall_s,
                    out.kernel_ms, out.launches, out.sweeps_per_launch);
        std::fflush(stdout);
        // CPU pool, matched wall
        const double W = out.wall_s;
        const int cap_runs = 1 << 18;
        std::vector<wi::RunOut> cres(static_cast<size_t>(cap_runs));
        std::vector<double> cwall(size_t(cap_runs), 0.0);
        std::atomic<int> next{0};
        const auto t0 = std::chrono::steady_clock::now();
        {
            std::vector<std::thread> pool;
            for (int t = 0; t < threads; ++t)
                pool.emplace_back([&]() {
                    for (;;) {
                        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                        if (el >= W) return;
                        const int k = next++;
                        if (k >= cap_runs) return;
                        const auto a0 = std::chrono::steady_clock::now();
                        cres[size_t(k)] = wi::anneal(g, b, uint64_t(k + 1));
                        cwall[size_t(k)] = std::chrono::duration<double>(std::chrono::steady_clock::now() - a0).count();
                    }
                });
            for (auto& th : pool) th.join();
        }
        const double Wc = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const int Rc = std::min(int(next.load()), cap_runs);
        std::printf("  CPU: %d runs on %d threads, wall %.3f s (matched to %.3f s)\n", Rc, threads, Wc, W);
        std::fflush(stdout);
        // verification, both devices
        std::vector<int64_t> gcut(static_cast<size_t>(R)), gsc(static_cast<size_t>(R)), ccut(static_cast<size_t>(Rc)), csc(static_cast<size_t>(Rc));
        std::vector<int> ident(size_t(R), -1);
        std::vector<std::string> gsha(static_cast<size_t>(R)), csha(static_cast<size_t>(Rc));
        const int shared_seeds = std::min(R, Rc);
        parallel_for(R, threads, [&](int r) {
            const std::vector<uint8_t> x = out.x(r);
            gcut[size_t(r)] = wi::cut_of(g, x);
            gsc[size_t(r)] = wi::energy_scratch(g, x);
            if (r < shared_seeds) {
                ident[size_t(r)] = x == cres[size_t(r)].x && out.energy[size_t(r)] == cres[size_t(r)].energy_tracked;
                gsha[size_t(r)] = wi::assignment_sha256(x);
            }
        });
        parallel_for(Rc, threads, [&](int r) {
            ccut[size_t(r)] = wi::cut_of(g, cres[size_t(r)].x);
            csc[size_t(r)] = wi::energy_scratch(g, cres[size_t(r)].x);
            csha[size_t(r)] = wi::assignment_sha256(cres[size_t(r)].x);
        });
        auto finish = [&](const char* dev, int runs, double wall, const std::vector<int64_t>& cut,
                          const std::vector<int64_t>& sc, auto etr, double latency) {
            Stat s;
            s.runs = runs;
            s.wall = wall;
            for (int r = 0; r < runs; ++r) {
                const bool cons = cut[size_t(r)] == -etr(r) && cut[size_t(r)] == -sc[size_t(r)];
                s.inconsistent += cons ? 0 : 1;
                const int ok = cut[size_t(r)] >= target ? 1 : 0;
                s.succ += ok;
                s.success.push_back(ok);
                s.sum_cut += double(cut[size_t(r)]);
                if (cut[size_t(r)] > s.best) { s.best = cut[size_t(r)]; s.best_seed = r + 1; }
            }
            const double upd = double(runs) * double(A + H) * double(g.n);
            const double rate = upd / wall;
            const double pp = runs ? double(s.succ) / runs : 0.0;
            const double t_am = runs ? wall / runs : 0.0;
            const double tts = wi::tts99(t_am, pp);
            const std::vector<double> times(size_t(runs), t_am);
            const wi::TtsCi ci = wi::tts99_bootstrap(s.success, times);
            const double gap = 100.0 * double(s.best - target) / double(target);
            std::fprintf(fs, "%s,%s,%d,%lld,%s,%s,%d,%d,%d,%s,%.0f,%s,%lld,%lld,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d\n", name,
                         dev, g.n, (long long)g.m, fmt(b.t_hot).c_str(), fmt(b.t_cold).c_str(), A, H, runs,
                         fmt(wall).c_str(), upd, fmt(rate).c_str(), (long long)s.best, target, fmt(gap).c_str(),
                         fmt(s.sum_cut / runs).c_str(), fmt(pp).c_str(), fmt(t_am).c_str(), fmt(tts).c_str(),
                         fmt(ci.lo).c_str(), fmt(ci.hi).c_str(), fmt(latency).c_str(),
                         fmt(wi::tts99(latency, pp)).c_str(), int(s.inconsistent == 0), s.best_seed);
            std::fflush(fs);
            std::printf("  %-3s runs %6d  best %lld (target %lld, gap %s%%)  mean %s  p %s  rate %.3e/s  TTS99 %s s [%s, %s]  consistent %s\n",
                        dev, runs, (long long)s.best, target, fmt(gap).c_str(), fmt(s.sum_cut / runs).c_str(),
                        fmt(pp).c_str(), rate, fmt(tts).c_str(), fmt(ci.lo).c_str(), fmt(ci.hi).c_str(),
                        s.inconsistent ? "NO" : "yes");
            std::fflush(stdout);
            return rate;
        };
        double lat_c = 0.0;
        for (int r = 0; r < Rc; ++r) lat_c += cwall[size_t(r)];
        lat_c = Rc ? lat_c / Rc : 0.0;
        const double rate_g = finish("gpu", R, out.wall_s, gcut, gsc, [&](int r) { return int64_t(out.energy[size_t(r)]); }, out.wall_s);
        const double rate_c = finish("cpu", Rc, Wc, ccut, csc, [&](int r) { return cres[size_t(r)].energy_tracked; }, lat_c);
        int identical = 0;
        for (int r = 0; r < shared_seeds; ++r) identical += ident[size_t(r)] == 1;
        std::printf("  speedup %.2fx; CPU/GPU identical %d of %d shared seeds; oracle mismatches %lld of 1000\n",
                    rate_g / rate_c, identical, shared_seeds, (long long)orc_bad);
        std::fflush(stdout);
        std::fprintf(ft, "%s,%s,%s,%s,%s,%s,%s,%s,%d,%d,%d,%d\n", name, plan_str(p).c_str(), fmt(rate_c).c_str(),
                     fmt(rate_g).c_str(), fmt(rate_g / rate_c).c_str(), fmt(out.wall_s).c_str(), fmt(Wc).c_str(),
                     fmt(out.kernel_ms).c_str(), out.launches, out.sweeps_per_launch, R, Rc);
        std::fprintf(fa, "%s,%d,%d,1000,%lld\n", name, shared_seeds, identical, (long long)orc_bad);
        std::fflush(ft);
        std::fflush(fa);
        for (int r = 0; r < R; ++r) {
            const bool cons = gcut[size_t(r)] == -out.energy[size_t(r)] && gcut[size_t(r)] == -gsc[size_t(r)];
            std::fprintf(fg, "%s,%d,%lld,%lld,%lld,%d,%d,%d,%s\n", name, r + 1, (long long)gcut[size_t(r)],
                         (long long)out.energy[size_t(r)], (long long)gsc[size_t(r)], int(cons),
                         int(gcut[size_t(r)] >= target), ident[size_t(r)], gsha[size_t(r)].c_str());
        }
        for (int r = 0; r < Rc; ++r) {
            const bool cons = ccut[size_t(r)] == -cres[size_t(r)].energy_tracked && ccut[size_t(r)] == -csc[size_t(r)];
            std::fprintf(fc, "%s,%d,%lld,%lld,%lld,%d,%d,%s,%s\n", name, r + 1, (long long)ccut[size_t(r)],
                         (long long)cres[size_t(r)].energy_tracked, (long long)csc[size_t(r)], int(cons),
                         int(ccut[size_t(r)] >= target), fmt(cwall[size_t(r)]).c_str(), csha[size_t(r)].c_str());
        }
        std::fflush(fg);
        std::fflush(fc);
    }
    std::fclose(list);
    std::fclose(fg); std::fclose(fc); std::fclose(fs); std::fclose(ft); std::fclose(fa); std::fclose(fl);
    std::printf("done\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--check") == 0) return mode_check(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "--bench") == 0) return mode_bench(argc, argv);
    std::fprintf(stderr, "usage: gpu_anneal --check [gset_dir] | --bench <list> <outdir> [R] [threads]\n");
    return 2;
}
