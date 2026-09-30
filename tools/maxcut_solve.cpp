// =============================================================================
// QuBLAR -- maxcut_solve: the integer engine on an external instance
// =============================================================================
//   maxcut_solve <graph.rudy> <seeds> <t_scale> <out.txt>
//
// Runs argos::wising::anneal (src/ising_weighted.hpp, ADR-015, unmodified) on a
// rudy graph for seeds 1..seeds and keeps the best cut. The schedule is the
// engine's default Budget (5 -> 1e-3, 400 + 20 sweeps) with both temperatures
// multiplied by t_scale: on a graph whose weights are not +-1 the schedule is
// carried in units of the instance's weight scale, which is how the G-set
// schedule transfers. Every run's tracked energy is checked against the cut
// counted from the edge list; a mismatch exits 2.
//
// Output file: line 1 "cut <best> seed <s>", line 2 the assignment as 0/1.
// Host C++17 only; first used by SUNET S6 (charging-point siting).
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../src/ising_weighted.hpp"

namespace wi = argos::wising;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: maxcut_solve <graph.rudy> <seeds> <t_scale> <out.txt>\n");
        return 2;
    }
    std::vector<char> buf;
    if (!wi::read_file(argv[1], buf)) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    wi::WeightedGraph g;
    std::string err;
    if (!wi::parse_rudy(buf.data(), buf.size(), g, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    const int seeds = std::atoi(argv[2]);
    const double scale = std::atof(argv[3]);
    wi::Budget b;
    b.t_hot *= scale;
    b.t_cold *= scale;

    int64_t best = std::numeric_limits<int64_t>::min();
    int best_seed = 0;
    std::vector<uint8_t> best_x;
    for (int s = 1; s <= seeds; ++s) {
        const wi::RunOut r = wi::anneal(g, b, uint64_t(s));
        const int64_t cut = wi::cut_of(g, r.x);
        if (cut != -r.energy_tracked) {
            std::fprintf(stderr, "seed %d: tracked energy %lld != -cut %lld\n", s,
                         (long long)r.energy_tracked, (long long)cut);
            return 2;
        }
        std::printf("seed %d cut %lld\n", s, (long long)cut);
        if (cut > best) { best = cut; best_seed = s; best_x = r.x; }
    }
    std::FILE* f = std::fopen(argv[4], "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", argv[4]); return 2; }
    std::fprintf(f, "cut %lld seed %d\n", (long long)best, best_seed);
    for (uint8_t v : best_x) std::fputc(v ? '1' : '0', f);
    std::fputc('\n', f);
    std::fclose(f);
    std::printf("best cut %lld (seed %d), n %d, m %lld\n", (long long)best, best_seed, g.n, (long long)g.m);
    return 0;
}
