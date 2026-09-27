// Gset MaxCut with the engine's own annealer. ADR-008: the QUBO is the interface.
//
// Schedule copied from check_ising's oracle (MAP), before any Gset number was
// read: t_hot = 5, t_cold = 1e-3, 400 anneal sweeps, 20 hold sweeps, 8 seeds.
// lambda = -1 so a disagreeing edge lowers the energy and the cut is -E.
// Edges must be weight +1. A negative weight is a different QUBO this binary
// does not pretend to solve.

#include "ising_recon.hpp"

#include <cstdio>
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

int main(int argc, char** argv) {
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
