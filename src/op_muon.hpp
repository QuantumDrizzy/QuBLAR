// =============================================================================
// QuBLAR -- sensor: cosmic-ray muons, transmission (ADR-006/007), as L1 rows
// =============================================================================
// One row per populated sky bin of each view. The row demands the deficit
// d = tau0 - t between the no-anomaly model and the measured optical depth,
// weighted by w = N_det (1 / Var(-ln T) for T << 1). A bit on the ray explains
// a = a_per_metre * segment length of it when it is 1.
// This is the code that lived inside build_binary_problem, unchanged.
// =============================================================================

#pragma once

#include "engine.hpp"
#include "muon_recon.hpp"

#include <limits>

namespace argos {

/// a_per_metre = NaN is the void path (a = mu_rock: a 1-bit removes rock). For
/// a denser body pass -(mu_body - mu_rock), from the declared density ratio.
inline OperatorRows muon_rows(const VoxelMedium& m_model, const std::vector<MuonView>& views,
                              const BitField& bits, double min_open = 30.0,
                              float a_per_metre = std::numeric_limits<float>::quiet_NaN()) {
    OperatorRows op;
    const float a_scale = std::isnan(a_per_metre) ? m_model.mu_rock : a_per_metre;
    for (const MuonView& view : views) {
        const MuonBinnedData& data = *view.data;
        for (int b = 0; b < data.size(); ++b) {
            if (double(data.open[b]) < min_open) continue;
            const float3 dir = data.bin_direction(b);
            const double tau0 = march_medium(m_model, view.chamber, dir);
            const double det = std::max(double(data.det[b]), 1.0);
            const double t = -std::log(det / double(data.open[b]));
            const int r = op.n_rows();
            bool touches = false;
            march_impl(m_model, view.chamber, dir, [&](int idx, float seg) {
                const int bit = bits.site_bit[idx];
                if (bit >= 0) {
                    op.entries.push_back({bit, r, a_scale * seg});
                    touches = true;
                }
                return 0.0f;
            });
            if (!touches) continue;
            op.d.push_back(tau0 - t);
            op.w.push_back(det);
        }
    }
    return op;
}

}  // namespace argos
