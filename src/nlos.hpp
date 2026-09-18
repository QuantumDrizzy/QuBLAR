// =============================================================================
// QuBLAR -- confocal backprojection: turning transients into a hidden volume
// =============================================================================
// ADR-004 items 6-7.
//
// For a confocal system the hidden points consistent with an arrival at time t from
// relay point p form a SPHERE of radius ct/2 - |p - s| centred on p. One relay point
// therefore says almost nothing: it places the object somewhere on a shell. Many relay
// points place it on many shells, and the shells intersect only where the object is.
//
// That is the whole idea, and it is the oldest NLOS reconstruction there is. Deliberately:
// it assumes nothing about the object beyond the transport model. Light cone transform and
// f-k migration are faster and sharper and both are inversions of this same forward model,
// so using one of them as the baseline would mean checking a model against itself.
// =============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "lidar.cuh"
#include "transient.cuh"

namespace argos {

/// The reconstruction grid and the system's temporal response must be MATCHED, and the
/// consequence of getting that wrong is not a blurry answer, it is a wrong one.
///
/// A voxel is scored by sampling the transient at the arrival time implied by its centre.
/// If the voxel is much coarser than the range resolution, the centre of the voxel
/// containing the object can be far enough off that its implied arrival time misses the
/// return completely -- and then no voxel lies on every shell, and the peak lands wherever
/// the sidelobes happen to agree.
///
/// It is easy to do by accident. At 4 ps bins one bin is 0.6 mm of range; a 2.5 cm voxel
/// is off by up to 20 of them while the pulse spans 3. The first attempt here did exactly
/// that and reconstructed an object 49 cm from where it was, with the physics entirely
/// correct underneath.
///
/// The rule: half the voxel diagonal, doubled for the round trip, must sit inside the
/// pulse width. `grid_to_pulse_ratio` below states it in code so it cannot drift.
struct Voxels {
    float3 lo, hi;
    int nx, ny, nz;
    std::vector<float> value;

    float3 center(int ix, int iy, int iz) const {
        return make_float3(lo.x + (hi.x - lo.x) * (ix + 0.5f) / nx,
                           lo.y + (hi.y - lo.y) * (iy + 0.5f) / ny,
                           lo.z + (hi.z - lo.z) * (iz + 0.5f) / nz);
    }
    size_t index(int ix, int iy, int iz) const {
        return (static_cast<size_t>(iz) * ny + iy) * nx + ix;
    }
};

/// Largest voxel half-diagonal, in metres -- the worst-case distance from a point in the
/// volume to the centre of the voxel holding it.
inline double voxel_half_diagonal(const Voxels& v) {
    const double dx = (v.hi.x - v.lo.x) / v.nx;
    const double dy = (v.hi.y - v.lo.y) / v.ny;
    const double dz = (v.hi.z - v.lo.z) / v.nz;
    return 0.5 * std::sqrt(dx * dx + dy * dy + dz * dz);
}

/// Ratio of the worst-case path error from voxel quantisation to the pulse sigma. Below
/// about 1 the grid is matched; far above it, the reconstruction is sampling noise.
inline double grid_to_pulse_ratio(const Voxels& v, const SensorConfig& cfg) {
    const double path_err = 2.0 * voxel_half_diagonal(v);       // out and back
    const double pulse_path = pulse_sigma(cfg) * kSpeedOfLight;
    return path_err / pulse_path;
}

/// Laplacian along the time axis, applied per relay point before backprojection.
///
/// Plain backprojection blurs: every voxel on a shell gets a vote, so the result is the
/// object convolved with the shell geometry, and the peak sits inside a large smooth blob
/// whose position is easy to bias. The second derivative in time is the standard filter
/// and it sharpens the shell into a thin one before any votes are cast.
///
/// It is applied here rather than folded into the forward model, so that the unfiltered
/// result stays available and the difference between them is visible rather than assumed.
inline std::vector<float> laplacian_filter(const std::vector<float>& tr, int n_relays, int bins) {
    std::vector<float> out(tr.size(), 0.f);
    for (int r = 0; r < n_relays; ++r) {
        const float* in = &tr[static_cast<size_t>(r) * bins];
        float* o = &out[static_cast<size_t>(r) * bins];
        for (int k = 1; k < bins - 1; ++k)
            o[k] = -in[k - 1] + 2.0f * in[k] - in[k + 1];
    }
    return out;
}

/// Vote every (relay point, time bin) onto the sphere it implies.
///
/// Gated: bins before `t_gate` are discarded. The first-bounce return off the relay
/// wall is orders of magnitude stronger than anything from the hidden object and would
/// bury it completely. Real systems gate it out too, which is why it is simulated rather
/// than suppressed at the source.
///
/// `l1_override` (ADR-005): when non-null, its entry r replaces |p - sensor| as the
/// sensor-to-wall leg of the path. Released confocal data is pre-rectified -- each
/// pixel's histogram was already shifted by its own 2|p-s|/c -- so for it the leg is
/// zero and the shells are centred on the relay points themselves. Passing 0 there is
/// not a special case invented for convenience; it is what their calibration did.
/// Null means "compute from the sensor position", which is every pre-existing caller.
inline void backproject(const std::vector<float>& transients,
                        const std::vector<RelayPoint>& relays,
                        float3 sensor, const SensorConfig& cfg,
                        Voxels& vox, float t_gate_seconds,
                        const std::vector<float>* l1_override = nullptr)
{
    const double c = kSpeedOfLight;
    const int bins = cfg.bins;
    vox.value.assign(static_cast<size_t>(vox.nx) * vox.ny * vox.nz, 0.f);

    std::vector<double> L1(relays.size());
    for (size_t r = 0; r < relays.size(); ++r) {
        if (l1_override) { L1[r] = (*l1_override)[r]; continue; }
        const float3 d = relays[r].position - sensor;
        L1[r] = std::sqrt(double(d.x) * d.x + double(d.y) * d.y + double(d.z) * d.z);
    }

    for (int iz = 0; iz < vox.nz; ++iz)
    for (int iy = 0; iy < vox.ny; ++iy)
    for (int ix = 0; ix < vox.nx; ++ix) {
        const float3 v = vox.center(ix, iy, iz);
        double acc = 0.0;
        for (size_t r = 0; r < relays.size(); ++r) {
            const float3 d = v - relays[r].position;
            const double rr = std::sqrt(double(d.x) * d.x + double(d.y) * d.y + double(d.z) * d.z);
            const double t = 2.0 * (L1[r] + rr) / c;
            if (t < t_gate_seconds) continue;
            const double fk = (t - cfg.t0_seconds) / cfg.bin_seconds - 0.5;
            const int k = static_cast<int>(std::floor(fk));
            if (k < 0 || k + 1 >= bins) continue;
            // Linear interpolation between bins. Snapping to the nearest bin quantises
            // every shell to the range bin width, and the intersection of quantised shells
            // is a lattice artefact rather than the object.
            const double f = fk - k;
            const float* tr = &transients[r * bins];
            acc += (1.0 - f) * tr[k] + f * tr[k + 1];
        }
        vox.value[vox.index(ix, iy, iz)] = static_cast<float>(acc);
    }
}

/// Location of the strongest voxel. The reconstruction is scored on distance from this to
/// the known object, in metres -- not a correlation and not an overlap score. This is a
/// localisation task, and a metric that can look good while the position is wrong would
/// defeat the point of having truth at all.
inline float3 peak_voxel(const Voxels& vox) {
    size_t best = 0;
    for (size_t i = 1; i < vox.value.size(); ++i)
        if (vox.value[i] > vox.value[best]) best = i;
    const int nx = vox.nx, ny = vox.ny;
    const int ix = static_cast<int>(best % nx);
    const int iy = static_cast<int>((best / nx) % ny);
    const int iz = static_cast<int>(best / (static_cast<size_t>(nx) * ny));
    return vox.center(ix, iy, iz);
}

}  // namespace argos
