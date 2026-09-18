// =============================================================================
// QuBLAR -- phasor-field reconstruction by angular-spectrum propagation
// =============================================================================
// ADR-005 item 4. The phasor field (Liu et al., "Non-line-of-sight imaging
// using phasor-field virtual wave optics", Nature 569, 2019) treats the gated
// wall transient as a time-varying virtual field on the wall, propagates that
// field into the hidden volume as a wave, and images where it focuses.
//
// The form used here, stated so the approximations are visible:
//
//   1. Per relay point, FFT the transient over time and keep a BAND of
//      temporal frequencies around f0 (positive frequencies only; the
//      transient is real, so the conjugate half carries no new information).
//   2. Divide each band component by i*omega (phase-averaged depth: the
//      component of the return that spreads along z is thereby given a phase
//      ramp in z -- Liu et al., supplementary; it is what lets one
//      propagation grid serve all depths).
//   3. Per frequency, FFT 2D over the wall and multiply by the Rayleigh-
//      Sommerfeld propagator exp(i k_z z) with k_z = sqrt((omega/c)^2 - kx^2
//      - ky^2); evanescent components decay as exp(-kappa z) instead.
//   4. Sum the band coherently and IFFT 2D per depth; |U|^2 is the volume.
//
// The virtual wavelength is not a free parameter to tune silently: f0 =
// 1/(2 pi sigma) sits where the Gaussian system response is down 4.4 dB, so
// lambda_v = 2 pi c sigma -- 8 cm for a 100 ps response, about a ninth of the
// released wall patch. `band_spacing` and `n_freqs` ARE exposed, because the
// released data does not carry the rig's true impulse response and the band
// choice is a documented sensitivity, not a fact about the hardware.
//
// Scale is normalised to the band maximum: only peak POSITIONS are scored,
// never amplitudes across methods.
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

struct PhasorConfig {
    int    n_freqs = 7;         // odd: a centre band plus symmetric sidebands
    double band_spacing = 0.35; // fractional spacing of sidebands around f0
    int    nz = 64;             // depth slices over [0, range/2]
    int    pad = 4;             // spatial zero-pad factor for propagation
    float  z_max_m = 0.f;       // scoring window over [0, z_max]; 0 = full range
};

/// 2D FFT over a p x p grid stored row-major. Separable, like fft3.
inline void fft2_inplace(std::vector<Cplx>& g, int p) {
    for (int y = 0; y < p; ++y)
        fft_inplace(&g[size_t(y) * p], size_t(p), false);
    std::vector<Cplx> col(p);
    for (int x = 0; x < p; ++x) {
        for (int y = 0; y < p; ++y) col[y] = g[size_t(y) * p + x];
        fft_inplace(col.data(), size_t(p), false);
        for (int y = 0; y < p; ++y) g[size_t(y) * p + x] = col[y];
    }
}

inline void phasor_reconstruct(const ExternalVolume& v, const PhasorConfig& cfg,
                               Voxels& out, double pulse_sigma_s) {
    const int n = v.n_grid, m = v.bins;
    const double c = kSpeedOfLight;
    const double pi = 3.14159265358979323846;
    const int p = n * cfg.pad;                    // padded wall grid
    const double dl = 2.0 * v.width / n;          // wall sample spacing
    const double dk = 2.0 * pi / (p * dl);
    const double df = 1.0 / (m * v.bin_seconds);
    const double f0 = 1.0 / (2.0 * pi * pulse_sigma_s);
    const int half = m / 2 + 1;                   // positive-frequency bins

    // --- temporal FFT, once per relay point ----------------------------------
    // spec[r*half + k] = positive-frequency spectrum of relay r. Computed once:
    // the naive alternative recomputes the full transform per band, seven
    // times, for a 64x64 grid -- the kind of cost that hides in a nested loop.
    std::vector<Cplx> spec(size_t(n) * n * half);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            std::vector<Cplx> row(m);
            const float* tr = &v.transients[(size_t(y) * n + x) * m];
            for (int t = 0; t < m; ++t) row[t] = Cplx(double(tr[t]), 0.0);
            fft_inplace(row.data(), size_t(m), false);
            Cplx* dst = &spec[(size_t(y) * n + x) * half];
            for (int k = 0; k < half; ++k) dst[k] = row[k];
        }

    // --- band selection, 1/(i omega), per-band normalisation -----------------
    std::vector<std::vector<Cplx>> f_omega;       // per band: FFT2 of the wall
    std::vector<double> omegas;
    for (int b = 0; b < cfg.n_freqs; ++b) {
        const int side = b - cfg.n_freqs / 2;
        const double f = f0 * (1.0 + side * cfg.band_spacing);
        const int kbin = static_cast<int>(std::lround(f / df));
        if (kbin <= 0 || kbin >= half) continue;  // band clipped by the gate
        const double omega = 2.0 * pi * kbin * df;

        // gamma(x,y) = 2 * R(x,y,omega_k) / (i*omega_k), zero-padded to p x p.
        // The factor 2 restores the negative-frequency half of the real field.
        std::vector<Cplx> g(size_t(p) * p, Cplx(0.0, 0.0));
        double gmax = 0.0;
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const Cplx r = spec[(size_t(y) * n + x) * half + kbin] * 2.0;
                const Cplx val = r / Cplx(0.0, omega);
                g[size_t(y) * p + x] = val;
                gmax = std::max(gmax, std::abs(val));
            }
        // Per-band normalisation to the strongest relay point. Bands differ in
        // raw amplitude by the system response; unnormalised, the coherent sum
        // would be dominated by whichever sideband sits nearest the response
        // peak, and the "band" would quietly be a band of one.
        if (gmax <= 0.0) continue;
        const double inv = 1.0 / gmax;
        for (auto& q : g) q *= inv;
        fft2_inplace(g, p);
        f_omega.push_back(std::move(g));
        omegas.push_back(omega);
    }

    // --- per-depth coherent propagation --------------------------------------
    // Spatial FFT indexing: bin l holds k = (l < p/2 ? l : l - p) * dk.
    out = Voxels{};
    out.nx = out.ny = n;
    out.nz = cfg.nz;
    out.lo = make_float3(float(-v.width), float(-v.width), 0.0f);
    const float z_top = (cfg.z_max_m > 0.f)
        ? std::min(float(0.5 * m * v.bin_seconds * c), cfg.z_max_m)
        : float(0.5 * m * v.bin_seconds * c);
    out.hi = make_float3(float(v.width), float(v.width), z_top);
    out.value.assign(size_t(out.nz) * n * n, 0.f);

    std::vector<Cplx> acc(size_t(p) * p);
    const double dz = (out.hi.z - out.lo.z) / cfg.nz;
    for (int iz = 0; iz < cfg.nz; ++iz) {
        const double z = (iz + 0.5) * dz;         // voxel centres, as Voxels does
        std::fill(acc.begin(), acc.end(), Cplx(0.0, 0.0));
        for (size_t b = 0; b < f_omega.size(); ++b) {
            const double k = omegas[b] / c;
            const std::vector<Cplx>& f = f_omega[b];
            for (int ly = 0; ly < p; ++ly) {
                const double ky = (ly < p / 2 ? ly : ly - p) * dk;
                for (int lx = 0; lx < p; ++lx) {
                    const double kx = (lx < p / 2 ? lx : lx - p) * dk;
                    const double k2 = k * k - kx * kx - ky * ky;
                    Cplx prop;
                    if (k2 > 0.0) {
                        const double kz = std::sqrt(k2);
                        // e^{-i k_z z}, not e^{+i k_z z}: the temporal FFT here
                        // synthesises e^{+i omega t} components, so forward
                        // propagation to +z carries e^{-i k_z z}. With the
                        // other sign the image forms behind the wall and the
                        // window fills with the edge of a focus that is never
                        // inside it.
                        prop = Cplx(std::cos(kz * z), -std::sin(kz * z));
                    } else {
                        prop = Cplx(std::exp(-std::sqrt(-k2) * z), 0.0);
                    }
                    const Cplx q = f[size_t(ly) * p + lx];
                    acc[size_t(ly) * p + lx] +=
                        Cplx(q.real() * prop.real() - q.imag() * prop.imag(),
                             q.real() * prop.imag() + q.imag() * prop.real());
                }
            }
        }
        // The field is recovered by inverse transform; the physical aperture is
        // cropped back out of the padded grid.
        std::vector<Cplx> wall = acc;
        fft2_inplace(wall, p);
        const int o = (p - n) / 2;
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const Cplx u = wall[size_t(y + o) * p + x + o];
                out.value[(size_t(iz) * n + y) * n + x] =
                    static_cast<float>(u.real() * u.real() + u.imag() * u.imag());
            }
    }
}

}  // namespace argos
