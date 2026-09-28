// =============================================================================
// QuBLAR -- sensor: gravimetry, as L1 rows (ADR-017)
// =============================================================================
// Stations on the surface measure the vertical gravity anomaly of a density
// contrast below them. A 1-bit is a cell of volume V carrying contrast drho;
// seen from a station it is a point mass (the cell's own centre):
//
//   a_sb = G * drho * V * (z_s - z_b) / r_sb^3        [microGal]
//
// with z up, so a denser body below a station gives a positive anomaly. Each
// row is one station: d = its observed anomaly, w = 1 / sigma^2 (Gaussian noise
// of declared sigma), so 1/2 w (A x - d)^2 is the negative log-likelihood in
// nats. Gravity is linear in the bits; it constrains mass well and depth
// poorly, a physical non-uniqueness the tri-state map is expected to show.
//
// Constants: G = 6.67430e-11 m^3 kg^-1 s^-2 (CODATA 2018). 1 microGal = 1e-8 m/s^2.
// Closed form checked in check_engine: a uniform sphere, outside it, attracts
// as a point mass at its centre (Newton's shell theorem).
// =============================================================================

#pragma once

#include "engine.hpp"

#include <cmath>
#include <vector>

namespace argos {

constexpr double kG = 6.67430e-11;       // m^3 kg^-1 s^-2
constexpr double kMicroGal = 1e-8;       // m / s^2

/// A regular grid of cells: cell (i, j, k) has centre lo + (i + 1/2, j + 1/2, k + 1/2) * h.
struct CellGrid {
    double lo[3] = {0, 0, 0};
    double h = 1.0;
    int nx = 0, ny = 0, nz = 0;

    void centre(int site, double c[3]) const {
        const int i = site % nx, j = (site / nx) % ny, k = site / (nx * ny);
        c[0] = lo[0] + (i + 0.5) * h;
        c[1] = lo[1] + (j + 0.5) * h;
        c[2] = lo[2] + (k + 0.5) * h;
    }
};

struct GravityStation {
    double p[3];
    double g_obs = 0.0;   // observed anomaly, microGal
};

/// Vertical anomaly at station s of a point mass m at c, in microGal.
inline double point_mass_gz(const double s[3], const double c[3], double m) {
    const double dx = s[0] - c[0], dy = s[1] - c[1], dz = s[2] - c[2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    return kG * m * dz / (r2 * std::sqrt(r2)) / kMicroGal;
}

inline OperatorRows gravity_rows(const CellGrid& g, const BitField& bits,
                                 const std::vector<GravityStation>& stations,
                                 double drho_kg_m3, double sigma_microgal) {
    OperatorRows op;
    const double cell_mass = drho_kg_m3 * g.h * g.h * g.h;
    for (int s = 0; s < int(stations.size()); ++s) {
        op.d.push_back(stations[s].g_obs);
        op.w.push_back(1.0 / (sigma_microgal * sigma_microgal));
        for (int b = 0; b < bits.n_bits(); ++b) {
            double c[3];
            g.centre(bits.site[b], c);
            op.entries.push_back({b, s, float(point_mass_gz(stations[s].p, c, cell_mass))});
        }
    }
    return op;
}

}  // namespace argos
