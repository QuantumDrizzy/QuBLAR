// =============================================================================
// QuBLAR -- a radix-2 FFT, written rather than linked
// =============================================================================
// ADR-005 item 2. The LCT and the phasor field both need FFTs and this
// repository links nothing outside CUDA and OptiX, which is a rule worth more
// than a fast transform. Iterative Cooley-Tukey, in-place, powers of two only
// -- asserted, not silently handled, because every size in this pipeline is a
// power of two by construction (64/128/256/512/1024).
//
// The convention matches MATLAB and numpy exactly: forward unnormalised,
// inverse scaled by 1/n. The LCT port is checked against a numpy mirror of the
// released MATLAB (tools/lct_reference.py), and a convention mismatch would
// surface there as total disagreement rather than as a small tolerance.
//
// Host only, on purpose (ADR-003: the yardstick is not measured on the GPU).
// =============================================================================

#pragma once

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace argos {

using Cplx = std::complex<double>;

/// True when n is a power of two. FFT sizes are asserted, because a radix-2
/// transform silently given a non-power-of-two length produces garbage that
/// still looks like a plausible spectrum.
inline bool fft_size_ok(std::size_t n) { return n > 0 && (n & (n - 1)) == 0; }

/// In-place iterative radix-2 FFT. `inverse` scales by 1/n, per the MATLAB and
/// numpy convention the port is checked against.
inline void fft_inplace(Cplx* a, std::size_t n, bool inverse) {
    assert(fft_size_ok(n));
    // Bit-reversal permutation. Swapping pairs in place is the standard
    // formulation; a table would be faster and is not worth the state.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const double sign = inverse ? 1.0 : -1.0;
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = sign * 2.0 * 3.14159265358979323846 / double(len);
        const Cplx wlen(std::cos(ang), std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {
            Cplx w(1.0, 0.0);
            for (std::size_t k = 0; k < len / 2; ++k) {
                const Cplx u = a[i + k];
                const Cplx v = a[i + k + len / 2] * w;
                a[i + k]           = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        const double inv = 1.0 / double(n);
        for (std::size_t i = 0; i < n; ++i) a[i] *= inv;
    }
}

/// FFT of a strided sequence of length d0 with stride `stride` inside `vol`,
/// used for batched multi-dimensional transforms: a 3D transform is three
/// separable passes of 1D ones over lines of the flattened volume.
inline void fft_lines(Cplx* vol, int d0, std::size_t stride, bool inverse) {
    std::vector<Cplx> line(d0);
    for (int i = 0; i < d0; ++i) line[i] = vol[std::size_t(i) * stride];
    fft_inplace(line.data(), std::size_t(d0), inverse);
    for (int i = 0; i < d0; ++i) vol[std::size_t(i) * stride] = line[i];
}

/// Separable 3D FFT over a [d0][d1][d2] volume stored row-major
/// (index = (i0*d1 + i1)*d2 + i2). Dimensions must be powers of two.
inline void fft3_inplace(std::vector<Cplx>& vol, int d0, int d1, int d2,
                         bool inverse) {
    assert(fft_size_ok(std::size_t(d0)));
    assert(fft_size_ok(std::size_t(d1)));
    assert(fft_size_ok(std::size_t(d2)));
    assert(vol.size() == std::size_t(d0) * d1 * d2);
    const std::size_t n1 = std::size_t(d1) * d2;
    for (int i0 = 0; i0 < d0; ++i0)                    // along dim 2 (contiguous)
        for (int i1 = 0; i1 < d1; ++i1)
            fft_inplace(&vol[i0 * n1 + std::size_t(i1) * d2], std::size_t(d2), inverse);
    std::vector<Cplx> col(d1);
    for (int i0 = 0; i0 < d0; ++i0)                    // along dim 1
        for (int i2 = 0; i2 < d2; ++i2) {
            for (int i1 = 0; i1 < d1; ++i1) col[i1] = vol[i0 * n1 + std::size_t(i1) * d2 + i2];
            fft_inplace(col.data(), std::size_t(d1), inverse);
            for (int i1 = 0; i1 < d1; ++i1) vol[i0 * n1 + std::size_t(i1) * d2 + i2] = col[i1];
        }
    std::vector<Cplx> col0(d0);
    for (int i1 = 0; i1 < d1; ++i1)                    // along dim 0 (strided)
        for (int i2 = 0; i2 < d2; ++i2) {
            const std::size_t base = std::size_t(i1) * d2 + i2;
            for (int i0 = 0; i0 < d0; ++i0) col0[i0] = vol[i0 * n1 + base];
            fft_inplace(col0.data(), std::size_t(d0), inverse);
            for (int i0 = 0; i0 < d0; ++i0) vol[i0 * n1 + base] = col0[i0];
        }
}

}  // namespace argos
