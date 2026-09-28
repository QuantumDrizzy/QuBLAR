// =============================================================================
// QuBLAR -- the engine's model layer, L1 (ADR-017)
// =============================================================================
// The unknowns are bits. A bit lives at a site (a grid cell, a graph node); the
// site is where it lives, not what it is. Three plug-ins build one problem:
//
//   BitField       the bits, the site each one lives at, and who touches whom
//   OperatorRows   what one sensor says: measurements (d, w) and each bit's
//                  share a of each measurement, with a bit set to 1
//   IsingPrior     rare (kappa per 1-bit) and continuous (lambda per unlike pair)
//
// assemble() stacks the rows of any number of sensors over one BitField and
// returns the BinaryProblem the annealer, the freezer and the ROI already use:
//
//   E(x) = 1/2 sum_rows w (sum_bits a x - d)^2 + lambda sum_<uv>[x_u != x_v]
//        + kappa sum x
//
// Two sensors on one field is the same code as one: their rows are stacked.
// =============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace argos {

struct BinaryProblem {
    // variables: the bits, in site order
    std::vector<int> var_voxel;       // bit -> site index
    std::vector<int> voxel_var;       // site -> bit, or -1
    // rows: every measurement of every sensor
    std::vector<double> d, w;
    // CSR: bit -> (row, a)
    std::vector<int> row_ptr, ray;
    std::vector<float> a;
    // adjacency among bits, CSR
    std::vector<int> nbr_ptr, nbr;
    double lambda = 0.0, kappa = 0.0;
    // Optional per-bit linear term replacing kappa: kappa plus lambda per
    // neighbour frozen at 0 (a bond to a frozen 0 costs lambda iff x = 1).
    std::vector<double> field;

    int n_vars() const { return int(var_voxel.size()); }
    double lin(int i) const { return field.empty() ? kappa : field[i]; }
    int n_rays() const { return int(d.size()); }
};

/// The unknowns: which sites carry a bit, and the adjacency the prior couples.
struct BitField {
    std::vector<int> site;       // bit -> site
    std::vector<int> site_bit;   // site -> bit, or -1
    std::vector<int> nbr_ptr, nbr;

    int n_bits() const { return int(site.size()); }

    /// Bits on the cells of an nx*ny*nz grid where domain[cell] != 0, with
    /// 6-neighbour adjacency, cells in x-fastest order.
    static BitField grid(int nx, int ny, int nz, const std::vector<char>& domain) {
        BitField f;
        const size_t n = size_t(nx) * ny * nz;
        f.site_bit.assign(n, -1);
        for (size_t c = 0; c < n; ++c)
            if (domain[c]) {
                f.site_bit[c] = int(f.site.size());
                f.site.push_back(int(c));
            }
        f.nbr_ptr.assign(f.n_bits() + 1, 0);
        for (int i = 0; i < f.n_bits(); ++i) {
            const int c = f.site[i];
            const int x = c % nx, y = (c / nx) % ny, z = c / (nx * ny);
            const int cand[6][3] = {{x - 1, y, z}, {x + 1, y, z}, {x, y - 1, z},
                                    {x, y + 1, z}, {x, y, z - 1}, {x, y, z + 1}};
            for (const auto& q : cand) {
                if (q[0] < 0 || q[0] >= nx || q[1] < 0 || q[1] >= ny || q[2] < 0 || q[2] >= nz)
                    continue;
                const int u = f.site_bit[(size_t(q[2]) * ny + q[1]) * nx + q[0]];
                if (u >= 0) f.nbr.push_back(u);
            }
            f.nbr_ptr[i + 1] = int(f.nbr.size());
        }
        return f;
    }
};

/// What one sensor says about a BitField. Rows are measurements; entries are
/// (bit, row, a): the share of row `row` that bit `bit` explains when it is 1.
struct OperatorRows {
    struct Entry { int bit, row; float a; };
    std::vector<double> d, w;
    std::vector<Entry> entries;

    int n_rows() const { return int(d.size()); }

    /// y = A x, for x in {0,1}^n or any real vector (the sensor's forward model).
    std::vector<double> apply(const std::vector<double>& x) const {
        std::vector<double> y(n_rows(), 0.0);
        for (const Entry& e : entries) y[e.row] += double(e.a) * x[e.bit];
        return y;
    }
    /// g = A^T r (the sensor's adjoint).
    std::vector<double> adjoint(const std::vector<double>& r, int n_bits) const {
        std::vector<double> g(n_bits, 0.0);
        for (const Entry& e : entries) g[e.bit] += double(e.a) * r[e.row];
        return g;
    }
};

struct IsingPrior {
    double lambda = 0.0;   // cost of an unlike neighbour pair
    double kappa = 0.0;    // cost of a 1-bit: ln((1 - p0) / p0) for prior fraction p0
};

/// Stack the rows of every sensor over one field, add the prior, and build the
/// problem the inference layer runs on. Entry order within a bit follows the
/// same sort as before L1, so a single-sensor problem is identical, array for
/// array, to what build_binary_problem produced.
inline BinaryProblem assemble(const BitField& f, const std::vector<const OperatorRows*>& sensors,
                              const IsingPrior& prior) {
    BinaryProblem p;
    p.lambda = prior.lambda;
    p.kappa = prior.kappa;
    p.var_voxel = f.site;
    p.voxel_var = f.site_bit;
    struct Entry { int var, ray; float a; };
    std::vector<Entry> entries;
    for (const OperatorRows* s : sensors) {
        const int offset = p.n_rays();
        p.d.insert(p.d.end(), s->d.begin(), s->d.end());
        p.w.insert(p.w.end(), s->w.begin(), s->w.end());
        for (const auto& e : s->entries) entries.push_back({e.bit, e.row + offset, e.a});
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
    p.nbr_ptr = f.nbr_ptr;
    p.nbr = f.nbr;
    return p;
}

/// The operator refusal (ADR-017 L1): <A x, r> must equal <x, A^T r> for random
/// x and r. Returns the relative mismatch; a sensor above tolerance does not load.
/// For rows stored explicitly this guards the apply/adjoint code paths; it bites
/// on matrix-free sensors. A sensor's physics is checked against its own closed
/// form, separately.
inline double adjoint_mismatch(const OperatorRows& op, int n_bits, uint64_t seed = 7) {
    uint64_t s = seed;
    auto next = [&s] {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return double(s >> 11) * 0x1.0p-53 - 0.5;
    };
    std::vector<double> x(n_bits), r(op.n_rows());
    for (double& v : x) v = next();
    for (double& v : r) v = next();
    const std::vector<double> ax = op.apply(x);
    const std::vector<double> atr = op.adjoint(r, n_bits);
    double lhs = 0.0, rhs = 0.0, scale = 0.0;
    for (int j = 0; j < op.n_rows(); ++j) lhs += ax[j] * r[j];
    for (int i = 0; i < n_bits; ++i) {
        rhs += x[i] * atr[i];
        scale += std::fabs(x[i] * atr[i]);
    }
    return std::fabs(lhs - rhs) / std::max(scale, 1e-300);
}

}  // namespace argos
