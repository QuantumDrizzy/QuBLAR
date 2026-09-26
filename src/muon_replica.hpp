// =============================================================================
// QuBLAR -- the ScanPyramids replica, shared by every muon check
// =============================================================================
// One pyramid, one void, one set of chambers and one exposure routine, used by
// check_muon (ADR-006) and check_ising (ADR-007). Two copies of the replica
// would be two answers to "what was measured" -- the confound ADR-006 §3
// already refused for the marcher.
// =============================================================================

#pragma once

#include "muon.cuh"
#include "muon_recon.hpp"

#include <vector>

namespace argos {

// -----------------------------------------------------------------------------
// The replica: Khufu's pyramid, ScanPyramids scale
// -----------------------------------------------------------------------------

static const float kHeight = 139.0f;    // metres
static const float kHalfBase = 115.0f;  // metres
static const float kMuRock = 0.046f;    // 1/m: T(100 m) ~ 1% (ADR-006 §1)
static const float kDetZ = 40.0f;       // chamber elevation
static const float3 kChamber = make_float3(0.0f, 0.0f, kDetZ);

static const int kN = 128;              // 128^3 voxels of 2 m: a 256 m cube
static const float kVoxel = 2.0f;
static const float3 kLo = make_float3(-128.0f, -128.0f, -10.0f);

static const float kVoidHX = 1.2f;      // the Big Void: ~2 x 2 m cross-section,
static const float kVoidHY = 15.0f;     // >= 30 m long, above the Grand Gallery
static const float kVoidZ = 77.0f;

static float pyramid_mu(float x, float y, float z) {
    if (z < 0.0f || z > kHeight) return 0.0f;
    const float w = kHalfBase * (1.0f - z / kHeight);
    return (fabsf(x) <= w && fabsf(y) <= w) ? kMuRock : 0.0f;
}

/// Host medium: solid rock (the known chambers are deliberately absent --
/// ADR-006 §6) plus, optionally, the void.
static VoxelMedium make_medium(bool with_void, std::vector<float>& host_mu,
                               float* d_mu) {
    host_mu.assign(size_t(kN) * kN * kN, kMuRock * (1.2f / 2700.0f));  // air
    for (int k = 0; k < kN; ++k)
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kN; ++i) {
                const float x = kLo.x + (i + 0.5f) * kVoxel;
                const float y = kLo.y + (j + 0.5f) * kVoxel;
                const float z = kLo.z + (k + 0.5f) * kVoxel;
                size_t idx = (size_t(k) * kN + j) * kN + i;
                if (pyramid_mu(x, y, z) > 0.0f) host_mu[idx] = kMuRock;
                if (with_void && fabsf(x) <= kVoidHX && fabsf(y) <= kVoidHY
                    && fabsf(z - kVoidZ) <= 1.2f)
                    host_mu[idx] = kMuRock * (1.2f / 2700.0f);
            }
    VoxelMedium m;
    m.lo = kLo; m.voxel = kVoxel; m.nx = m.ny = m.nz = kN;
    m.mu_rock = kMuRock;
    m.mu_air = kMuRock * (1.2f / 2700.0f);
    m.mu = d_mu;
    CUDA_CHECK(cudaMemcpy(d_mu, host_mu.data(),
                          host_mu.size() * sizeof(float), cudaMemcpyHostToDevice));
    return m;
}

/// Expose one chamber to the sky through a device medium; return its counts.
static MuonBinnedData expose(const VoxelMedium& m_dev, float3 chamber,
                             unsigned seed, MuonSky sky, int n_cand) {
    unsigned long long *d_open = nullptr, *d_det = nullptr;
    CUDA_CHECK(cudaMalloc(&d_open, sky.size() * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_det, sky.size() * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_open, 0, sky.size() * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_det, 0, sky.size() * sizeof(unsigned long long)));
    muon_expose_binned<<<muon_grid(n_cand, 256), 256>>>(
        m_dev, sky, n_cand, chamber, seed, d_open, d_det);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MuonBinnedData out;
    out.sky = sky;
    out.open.resize(sky.size());
    out.det.resize(sky.size());
    CUDA_CHECK(cudaMemcpy(out.open.data(), d_open,
                          sky.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.det.data(), d_det,
                          sky.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_open); cudaFree(d_det);
    return out;
}

/// The imaging sky. The 48 x 20 sky of sections C-D checks the statistics;
/// it cannot image the void: bins of 7.5 x 3.5 deg are wider than the ~3.7 deg
/// the 2.4 m void subtends at 37 m, so no ray resolves it and MLEM spreads the
/// deficit evenly along each ray (measured: identical mu from z = 73 to 101 m).
/// 1 deg bins resolve it; the candidate count scales with the bin count so
/// each bin keeps ~1000 open muons.
static MuonSky imaging_sky() {
    MuonSky s;
    s.n_az = 360;
    s.n_th = 70;
    return s;
}
static const int kImagingCandidates = 1 << 25;

/// Chambers: the ScanPyramids-like one at (0, 0, 40), plus two offset in x
/// that see the void's layer from other angles. One view gives directions;
/// crossing views give depth.
static const float3 kChambers[3] = {make_float3(0.0f, 0.0f, kDetZ),
                                    make_float3(-40.0f, 0.0f, 25.0f),
                                    make_float3(40.0f, 0.0f, 25.0f)};

}  // namespace argos
