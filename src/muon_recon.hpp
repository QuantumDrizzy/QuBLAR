// =============================================================================
// QuBLAR -- muon reconstruction: binned transmission, MLEM, and the control
// =============================================================================
// ADR-006 item 2. Host-only, on purpose (the ADR-003 rule: the yardstick does
// not live where the thing it measures lives). Every ray walked here goes
// through march_impl -- the SAME stepping the GPU exposure used (ADR-006 §3).
//
// Data product: per direction bin, open counts N^open (the candidate
// population) and detected counts N^det (the shadow). Measurement model:
//
//     N^det ~ Binomial(N^open, exp(-tau_hat_b)),  tau_hat_b = sum_v mu_v l_bv
//
// where l_bv is the path of the bin-centre ray through voxel v (the chamber
// is point-like, so the direction fully determines the ray -- ADR-006 §2).
//
// The MLEM update is derived, not copied (ADR-006 §4):
//
//     mu_v <- mu_v * sum_b l_bv N^open_b e^{-tau_hat_b}
//                / sum_b l_bv N^det_b
//
// fixed point e^{-tau_hat_b} = N^det/N^open, and the direction is right in
// both failure regimes: a voxel set too high makes e^{-tau_hat} too small and
// shrinks itself; a bin observed darker than modelled (few detections) pushes
// the voxels along its ray UP through the numerator. The denominator is the
// detected-ray illumination map, a per-voxel constant. The control is a
// one-step least-squares backprojection -- the oldest inversion that assumes
// nothing beyond the line-integral model.
// =============================================================================

#pragma once

#include <cmath>
#include <vector>

#include "muon.cuh"

namespace argos {

/// Direction-binned exposure, brought back from the device. The counts are
/// integers on the device; they stay integers here. (An earlier draft made
/// these doubles and memcpy'd the raw u64 bits in: every count became a
/// denormal ~1e-320, every bin fell under the population floor, and the MLEM
/// updated nothing -- silently, because a zero is a valid count.)
struct MuonBinnedData {
    MuonSky sky;
    std::vector<unsigned long long> open, det;   // [n_th * n_az]

    int size() const { return sky.n_th * sky.n_az; }
    /// Central direction of a bin. The reconstruction marches these; the
    /// candidates that filled the bin marched their own jittered directions,
    /// and the difference is the histogramming approximation, declared.
    float3 bin_direction(int bin) const {
        const int ith = bin / sky.n_az, iaz = bin % sky.n_az;
        const float th = (ith + 0.5f) / sky.n_th * sky.th_max;
        const float az = (iaz + 0.5f) * 6.28318530718f / sky.n_az;
        const float s = sinf(th);
        return make_float3(s * cosf(az), s * sinf(az), cosf(th));
    }
};

/// One detector position and the counts it recorded. A single point-like
/// chamber measures DIRECTIONS only: a deficit along a ray could sit anywhere
/// on it. Depth comes from crossing rays, i.e. from more than one view --
/// which is how ScanPyramids localised the Big Void (several instruments,
/// several positions).
struct MuonView {
    float3 chamber;
    const MuonBinnedData* data;
};

/// Illuminated map: per voxel, the detected-ray path length. Voxels the
/// chamber never sees (or sees only through under-populated bins) are not
/// reconstructable and are excluded from scoring.
inline std::vector<float> muon_illumination(const VoxelMedium& m,
                                            float3 chamber,
                                            const MuonBinnedData& data,
                                            double min_open = 30.0) {
    std::vector<float> den(size_t(m.nx) * m.ny * m.nz, 0.0f);
    for (int b = 0; b < data.size(); ++b) {
        if (double(data.open[b]) < min_open) continue;
        const float3 d = data.bin_direction(b);
        march_impl(m, chamber, d, [&](int idx, float seg) {
            den[idx] += seg * float(data.det[b]);
            return 0.0f;   // recorders contribute nothing to the optical depth
        });
    }
    return den;
}

/// Transmission MLEM, in the form the muography literature uses (PET-style
/// update on the log-transmission data):
///
///     mu_v <- mu_v * sum_b l_bv w_b (-ln T_b)/tau_hat_b / sum_b l_bv w_b
///
/// with w_b = N^open_b. Fixed point: tau_hat_b = -ln T_b where the data says
/// so. Direction: a voxel set too high inflates tau_hat, shrinks the ratio,
/// and shrinks itself. And it is BOUNDED, which the naive ratio-of-sums form
/// this replaced was not: that one let voxels crossed by few, noisy bins run
/// to 1700x rock in 30 iterations while their neighbours collapsed to zero.
/// A bin with zero detections is clamped to -ln(1/N^open) = ln N^open: the
/// true implication is tau_hat >= ln N^open, and this is the mildest estimate
/// consistent with it.
///
/// The forward march runs over the ESTIMATE, never over the truth medium --
/// the first draft marched the true densities, which made the model already
/// know the answer and turned the iteration into a multiplicative random
/// walk on counting noise.
///
/// `mu_init` seeds the estimate and `domain` gates the update: voxels outside
/// the domain hold their prior forever. This is not a shortcut -- without it
/// the problem is massively underdetermined (one line integral, thousands of
/// free voxels including all of the air), and the maximum-likelihood solution
/// smears the pyramid's attenuation across the sky. Real muography recon-
/// structs the object it can see; the object's outer shape is surveying
/// knowledge, not the secret. `m_geom` contributes geometry only.
inline std::vector<float> mlem_transmission(const VoxelMedium& m_geom,
                                            const std::vector<MuonView>& views,
                                            int iters,
                                            const std::vector<float>& mu_init,
                                            const std::vector<char>& domain,
                                            double min_open = 30.0) {
    const size_t n_vox = size_t(m_geom.nx) * m_geom.ny * m_geom.nz;
    std::vector<float> mu_est = mu_init;
    VoxelMedium m_est = m_geom;
    m_est.mu = mu_est.data();       // the forward model marches the estimate
    std::vector<double> num(n_vox), den(n_vox);

    // The denominator is sum_b l_bv w_b over exactly the bins the numerator
    // uses, with the SAME weight w_b = N^open_b. The first cut divided by the
    // illumination map, which weights by N^det = N^open * T: the fixed point
    // then sat at tau_hat = -ln T / T, not at tau_hat = -ln T -- a bias of
    // about 1/T, i.e. ~100x through 100 m of rock.
    for (int it = 0; it < iters; ++it) {
        std::fill(num.begin(), num.end(), 0.0);
        std::fill(den.begin(), den.end(), 0.0);
        for (const MuonView& view : views)
        for (int b = 0; b < view.data->size(); ++b) {
            const MuonBinnedData& data = *view.data;
            const float3 chamber = view.chamber;
            if (double(data.open[b]) < min_open) continue;
            const float3 d = data.bin_direction(b);
            const float tau = march_medium(m_est, chamber, d);
            if (tau <= 0.0f) continue;   // open sky: carries no density signal
            const double t = -std::log(std::max(double(data.det[b]), 1.0)
                                       / double(data.open[b]));
            const double w = double(data.open[b]);
            const double ratio = t / double(tau);
            march_impl(m_est, chamber, d, [&](int idx, float seg) {
                num[idx] += w * seg * ratio;
                den[idx] += w * seg;
                return 0.0f;
            });
        }
        for (size_t v = 0; v < n_vox; ++v)
            if (domain[v] && den[v] > 0.0)
                mu_est[v] = static_cast<float>(double(mu_est[v]) * num[v]
                                               / den[v]);
    }
    return mu_est;
}

/// One-step least-squares backprojection, the control:
/// mu_v = sum_b N^open l_bv (-ln T_b) / sum_b N^open l_bv^2.
/// The same det=0 clamp as the MLEM: a fully absorbed bin claims tau >=
/// ln(N^open), and claiming zero thickness instead would call the darkest ray
/// the emptiest one.
/// Single-view convenience: one chamber, one data set.
inline std::vector<float> mlem_transmission(const VoxelMedium& m_geom,
                                            float3 chamber,
                                            const MuonBinnedData& data,
                                            int iters,
                                            const std::vector<float>& mu_init,
                                            const std::vector<char>& domain,
                                            double min_open = 30.0) {
    return mlem_transmission(m_geom, {MuonView{chamber, &data}}, iters,
                             mu_init, domain, min_open);
}

/// Illumination summed over views: a voxel is reconstructable if any view
/// crosses it with a populated bin.
inline std::vector<float> muon_illumination(const VoxelMedium& m,
                                            const std::vector<MuonView>& views,
                                            double min_open = 30.0) {
    std::vector<float> den(size_t(m.nx) * m.ny * m.nz, 0.0f);
    for (const MuonView& v : views) {
        const std::vector<float> one = muon_illumination(m, v.chamber, *v.data,
                                                         min_open);
        for (size_t i = 0; i < den.size(); ++i) den[i] += one[i];
    }
    return den;
}

inline std::vector<float> muon_backprojection(const VoxelMedium& m,
                                              float3 chamber,
                                              const MuonBinnedData& data,
                                              double min_open = 30.0) {
    const size_t n_vox = size_t(m.nx) * m.ny * m.nz;
    std::vector<double> num(n_vox, 0.0), den(n_vox, 0.0);
    for (int b = 0; b < data.size(); ++b) {
        if (double(data.open[b]) < min_open) continue;
        const float3 d = data.bin_direction(b);
        const double t = -std::log(std::max(double(data.det[b]), 1.0)
                                   / double(data.open[b]));
        march_impl(m, chamber, d, [&](int idx, float seg) {
            const double w = double(data.open[b]);
            num[idx] += w * seg * t;
            den[idx] += w * seg * seg;
            return 0.0f;
        });
    }
    std::vector<float> mu(n_vox);
    for (size_t v = 0; v < n_vox; ++v)
        mu[v] = den[v] > 0.0 ? static_cast<float>(num[v] / den[v]) : m.mu_rock;
    return mu;
}

/// Backprojection over several views: the same least-squares control, with
/// the sums running over every view's bins.
inline std::vector<float> muon_backprojection(const VoxelMedium& m,
                                              const std::vector<MuonView>& views,
                                              double min_open = 30.0) {
    const size_t n_vox = size_t(m.nx) * m.ny * m.nz;
    std::vector<double> num(n_vox, 0.0), den(n_vox, 0.0);
    for (const MuonView& view : views) {
        const MuonBinnedData& data = *view.data;
        for (int b = 0; b < data.size(); ++b) {
            if (double(data.open[b]) < min_open) continue;
            const float3 d = data.bin_direction(b);
            const double t = -std::log(std::max(double(data.det[b]), 1.0)
                                       / double(data.open[b]));
            const double w = double(data.open[b]);
            march_impl(m, view.chamber, d, [&](int idx, float seg) {
                num[idx] += w * seg * t;
                den[idx] += w * seg * seg;
                return 0.0f;
            });
        }
    }
    std::vector<float> mu(n_vox);
    for (size_t v = 0; v < n_vox; ++v)
        mu[v] = den[v] > 0.0 ? static_cast<float>(num[v] / den[v]) : m.mu_rock;
    return mu;
}

}  // namespace argos
