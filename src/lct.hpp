// =============================================================================
// QuBLAR -- the light cone transform, ported from the authors' released code
// =============================================================================
// ADR-005 item 4. A line-by-line port of cnlos_reconstruction.m, released with
// O'Toole, Lindell & Wetzstein, "Confocal Non-Line-of-Sight Imaging Based on
// the Light Cone Transform" (Nature 555, 2018). This is THEIR algorithm, not
// ours: it is here because a wave-based reconstruction that shares no code
// path with the forward model is the first real test of the ADR-004 claim that
// the baselines check assumptions rather than share them.
//
// The port is checked twice (ADR-005 item 5): numerically against a numpy
// mirror of the same MATLAB (tools/lct_reference.py, which dumps a golden
// volume), and against the paper's published figures visually.
//
// Conventions that must match the MATLAB exactly, because each one silently
// mirrors or shifts the volume if missed:
//   - their permute([3 2 1]) maps the rectified cube to [t, y, x];
//   - the PSF grid is 1-based in the original: the normalisation column is
//     index U = N (1-based) of a 2N grid, i.e. 0-based N-1, not the centre;
//   - the PSF is circshifted by U along both spatial axes;
//   - the t->sqrt(t) resampling operator is built at the DOWNSAMPLED size M
//     and then halved log2(M) times inside the operator itself, which is why
//     the closed form below sums 2^-K/sqrt(x) over x in [M*i, M*(i+1)).
// =============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "external.hpp"
#include "fft.hpp"
#include "nlos.hpp"

namespace argos {

struct LctConfig {
    float snr = 0.8f;        // their Wiener parameter; scene meta overrides for real data
    bool  diffuse = false;   // z^4 radiometric scaling instead of z^2
    float z_max_m = 0.f;     // scoring window: keep depth slices below this.
                             // 0 keeps the full [0, range/2]. The full volume's
                             // global maximum is the deconvolution's far-end DC
                             // blob -- the released pipeline crops it away for
                             // display, and so does any honest peak search.
};

/// The t->sqrt(t) resampling operator in closed form.
///
/// The released resamplingOperator(M) builds an (M^2 x M) matrix with
/// 1/sqrt(x) at column ceil(sqrt(x)) of row x, then averages adjacent rows
/// log2(M) times. After K halvings the final row i is exactly
///
///     a[i][c] = 2^-K * sum over x in [M*i, M*(i+1)) of 1/sqrt(x) at c = ceil(sqrt(x))-1
///
/// (2^K = M, so the windows are exactly M wide), and the "inverse" operator
/// is its transpose. Computing the closed form directly avoids materialising
/// the 262144 x 512 intermediate the MATLAB and the numpy mirror both build.
inline void resampling_operators(int m, std::vector<double>& a,
                                 std::vector<double>& at) {
    const int k = static_cast<int>(std::round(std::log2(double(m))));
    const double scale = std::ldexp(1.0, -k);          // 2^-K
    a.assign(size_t(m) * m, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int r = 0; r < m; ++r) {                  // 0-based original row
            const double x = double(i * m + r + 1);    // 1-based x
            const int c = static_cast<int>(std::ceil(std::sqrt(x))) - 1;
            a[size_t(i) * m + c] += scale / std::sqrt(x);
        }
    }
    at.assign(size_t(m) * m, 0.0);
    for (int i = 0; i < m; ++i)
        for (int c = 0; c < m; ++c)
            at[size_t(i) * m + c] = a[size_t(c) * m + i];
}

/// Run the released confocal LCT pipeline on a rectified ExternalVolume and
/// fill `out` with a PHYSICAL volume: index (iz*ny+iy)*nx+ix over
/// x,y in [-width, width], z in [0, range/2] -- directly comparable with the
/// backprojection Voxels and with truth. The MATLAB's display crop and x flip
/// are deliberately not applied here; they are presentation, not physics.
inline void lct_reconstruct(const ExternalVolume& v, const LctConfig& cfg,
                            Voxels& out) {
    const int n = v.n_grid, m = v.bins;
    const double c = kSpeedOfLight;
    const double bin_s = v.bin_seconds;
    const double rng = m * c * bin_s;
    const double slope = v.width / rng;

    // --- their permute([3 2 1]) + radiometric scaling ------------------------
    // data[t][y][x] = rect[x][y][t] * z_t^p,  z_t = t/(M-1)
    std::vector<double> data(size_t(m) * n * n);
    for (int x = 0; x < n; ++x)
        for (int y = 0; y < n; ++y)
            for (int t = 0; t < m; ++t) {
                const double z = (m > 1) ? double(t) / double(m - 1) : 0.0;
                const double g = cfg.diffuse ? z * z * z * z : z * z;
                data[(size_t(t) * n + y) * n + x] =
                    double(v.rect[(size_t(x) * n + y) * m + t]) * g;
            }

    // --- the resampling operator, then the padded transform volume -----------
    std::vector<double> a, at;
    resampling_operators(m, a, at);

    std::vector<Cplx> tdata(size_t(2 * m) * 2 * n * 2 * n, Cplx(0.0, 0.0));
    const size_t n1 = size_t(2 * n) * 2 * n;
    for (int t = 0; t < m; ++t)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                // (mtx * data)[t][y][x] = sum_k a[t][k] * data[k][y][x].
                // The write index is the PADDED grid's: row stride 2n, not n.
                // Filling with a flat j = y*n + x folds the spatial block 2:1
                // against how fft3 reads it -- the transform stays a transform,
                // the total sum is untouched (sums are permutation invariant),
                // and the result is a volume that peaks in the right place
                // while agreeing with nothing. Produced, measured, understood.
                double acc = 0.0;
                for (int k = 0; k < m; ++k)
                    acc += a[size_t(t) * m + k] * data[(size_t(k) * n + y) * n + x];
                tdata[size_t(t) * n1 + size_t(y) * (2 * n) + x] = Cplx(acc, 0.0);
            }

    // --- the light-cone PSF and its Wiener inverse ---------------------------
    // psf[iz][iy][ix] on the 2M x 2N x 2N grid, their 1-based quirks preserved.
    const double s4 = (4.0 * slope) * (4.0 * slope);
    std::vector<Cplx> psf(size_t(2 * m) * n1, Cplx(0.0, 0.0));
    std::vector<double> col_min(n1);                    // min over z, per (y,x).
                                                        // n1 is a variable, so
                                                        // parens are not a parse
                                                        // trap; braces would be
                                                        // a narrowing error.
    for (int iy = 0; iy < 2 * n; ++iy)
        for (int ix = 0; ix < 2 * n; ++ix) {
            const double gy = -1.0 + 2.0 * double(iy) / double(2 * n - 1);
            const double gx = -1.0 + 2.0 * double(ix) / double(2 * n - 1);
            const double rad = s4 * (gx * gx + gy * gy);
            double mn = 1e300;
            for (int iz = 0; iz < 2 * m; ++iz) {
                const double gz = 2.0 * double(iz) / double(2 * m - 1);
                const double val = std::fabs(rad - gz);
                psf[size_t(iz) * n1 + size_t(iy) * 2 * n + ix] = Cplx(val, 0.0);
                mn = std::min(mn, val);
            }
            col_min[size_t(iy) * 2 * n + ix] = mn;
        }
    // Binary shell, normalised by the sum over the (1-based) column U = N.
    double col_sum = 0.0;
    for (int iz = 0; iz < 2 * m; ++iz)
        if (psf[size_t(iz) * n1 + size_t(n - 1) * 2 * n + n - 1].real()
            == col_min[size_t(n - 1) * 2 * n + n - 1]) col_sum += 1.0;
    double norm2 = 0.0;
    for (int iz = 0; iz < 2 * m; ++iz)
        for (int iy = 0; iy < 2 * n; ++iy)
            for (int ix = 0; ix < 2 * n; ++ix) {
                Cplx& p = psf[size_t(iz) * n1 + size_t(iy) * 2 * n + ix];
                const double b = (p.real() == col_min[size_t(iy) * 2 * n + ix]) ? 1.0 : 0.0;
                p = Cplx(b / col_sum, 0.0);
                norm2 += p.real() * p.real();
            }
    const double s = 1.0 / (std::sqrt(norm2) * col_sum);
    // circshift by (0, N, N) on a 2N grid: TWO independent half-turns, one per
    // spatial axis, EACH applied to every z slice. The first attempt iterated
    // only (iy, ix) -- indexing without the iz term rolls slice zero and
    // leaves 1023 slices as an unnormalised binary shell, which preserves the
    // volume's total energy while scrambling everything the filter was for.
    // Each half-turn is an involution, so in-place pair swaps are safe; a
    // naive forward loop reads cells it has already overwritten and loses
    // half the shell.
    for (int iz = 0; iz < 2 * m; ++iz)
        for (int iy = 0; iy < n; ++iy)
            for (int ix = 0; ix < 2 * n; ++ix) {
                Cplx lo = psf[size_t(iz) * n1 + size_t(iy) * 2 * n + ix];
                Cplx hi = psf[size_t(iz) * n1 + size_t(iy + n) * 2 * n + ix];
                psf[size_t(iz) * n1 + size_t(iy + n) * 2 * n + ix] = Cplx(lo.real() * s, 0.0);
                psf[size_t(iz) * n1 + size_t(iy) * 2 * n + ix]     = Cplx(hi.real() * s, 0.0);
            }
    for (int iz = 0; iz < 2 * m; ++iz)
        for (int iy = 0; iy < 2 * n; ++iy)
            for (int ix = 0; ix < n; ++ix) {
                Cplx lo = psf[size_t(iz) * n1 + size_t(iy) * 2 * n + ix];
                Cplx hi = psf[size_t(iz) * n1 + size_t(iy) * 2 * n + ix + n];
                psf[size_t(iz) * n1 + size_t(iy) * 2 * n + ix + n] = lo;
                psf[size_t(iz) * n1 + size_t(iy) * 2 * n + ix]     = hi;
            }
    fft3_inplace(psf, 2 * m, 2 * n, 2 * n, false);
    const double reg = 1.0 / cfg.snr;
    for (auto& p : psf) {
        const double den = p.real() * p.real() + p.imag() * p.imag() + reg;
        p = Cplx(p.real() / den, -p.imag() / den);     // conj / (|F|^2 + 1/snr)
    }

    // --- convolve, unpad, resample the depth axis back -----------------------
    fft3_inplace(tdata, 2 * m, 2 * n, 2 * n, false);
    for (size_t i = 0; i < tdata.size(); ++i) {
        const Cplx f = tdata[i], h = psf[i];
        tdata[i] = Cplx(f.real() * h.real() - f.imag() * h.imag(),
                        f.real() * h.imag() + f.imag() * h.real());
    }
    fft3_inplace(tdata, 2 * m, 2 * n, 2 * n, true);

    std::vector<double> tvol(size_t(m) * n * n);
    for (int t = 0; t < m; ++t)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x)
                tvol[(size_t(t) * n + y) * n + x] =
                    tdata[size_t(t) * n1 + size_t(y) * (2 * n) + x].real();

    out = Voxels{};
    out.nx = out.ny = n;
    out.nz = m;
    out.lo = make_float3(float(-v.width), float(-v.width), 0.0f);
    out.hi = make_float3(float(v.width), float(v.width), float(0.5 * rng));
    out.value.assign(out.nz * out.ny * out.nx, 0.f);

    // vol[t][j] = sum_k at[t][k] * tvol[k][j], with at = a^T. The output is
    // already on the uniform depth axis z = linspace(0, rng/2, M): the resample
    // to sqrt(t) happened in the padded transform, and mtxi undoes exactly
    // that. No interpolation on top -- adding one silently shifts every slice.
    for (int j = 0; j < n * n; ++j) {
        const int iy = j / n, ix = j % n;
        for (int t = 0; t < m; ++t) {
            double acc = 0.0;
            for (int k = 0; k < m; ++k)
                acc += at[size_t(t) * m + k] * tvol[size_t(k) * n * n + j];
            out.value[(size_t(t) * n + iy) * n + ix] =
                static_cast<float>(std::max(acc, 0.0));
        }
    }

    // Declared scoring window: drop the slices above z_max. Slices are ordered
    // z-outermost, so truncation keeps the first nz_keep planes on the ORIGINAL
    // z grid, and hi.z shrinks to match so voxel centres stay truthful.
    if (cfg.z_max_m > 0.f) {
        const double slice = 0.5 * rng / m;
        int nz_keep = static_cast<int>(std::ceil(cfg.z_max_m / slice));
        nz_keep = std::max(1, std::min(nz_keep, m));
        out.value.resize(size_t(nz_keep) * n * n);
        out.nz = nz_keep;
        out.hi.z = float(nz_keep * slice);
    }
}

}  // namespace argos
