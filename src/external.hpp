// =============================================================================
// QuBLAR -- external confocal data, in the layout this repository already speaks
// =============================================================================
// ADR-005 item 3. Loads the converted captures of O'Toole, Lindell & Wetzstein
// (Nature 555, 2018) -- float32 dumps plus a key=value sidecar written by
// tools/mat_to_raw.py -- and emits exactly what the NLOS consumers expect:
// per-relay transients in the [relay * bins] layout of nlos.hpp, and the
// [x, y, t] rect cube in the order their pipeline uses.
//
// The same type also carries the SYNTHETIC REPLICA (check_external.cu): the
// replica is simulated, rectified the same way, and fed through the identical
// consumers, so a difference between synthetic and real results cannot come
// from the glue.
//
// Their data is pre-rectified: the direct wall return starts at the first time
// bin, per-pixel shifted, and cropped. The relay grid therefore lives on the
// wall plane z = 0 with the hidden volume at z > 0 -- their convention, not an
// arbitrary one, and it is why backproject gets an L1 override of zero for
// this data (see nlos.hpp).
// =============================================================================

#pragma once

#include "transient.cuh"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace argos {

/// Everything the reconstruction methods need, independent of where the data
/// came from. `rect` and `transients` are two views of the same numbers:
///   rect[idx]        with idx = (x*N + y)*M + t      (the released order)
///   transients[r*M+t] with r  = y*N + x              (the nlos.hpp order)
/// The x/y swap between them is real and load-bearing: the release indexes x
/// fastest, the relay grid indexes y fastest, and silently assuming they match
/// would mirror the whole scene.
struct ExternalVolume {
    int   n_grid = 0;        // N, relay points per axis
    int   bins = 0;          // M, after downsampling
    float bin_seconds = 0.f; // after downsampling
    float width = 0.f;       // wall half-extent, metres
    bool  diffuse = false;   // their diffuse flag: z^4 instead of z^2 scaling
    float snr = 0.8f;        // their per-scene Wiener parameter
    int   z_trim = 0;        // bins gated at t=0 (already applied)
    int   z_offset_display = 0;

    std::vector<float> rect;            // [N, N, M], gated
    std::vector<RelayPoint> relays;     // N*N, index = y*N + x, on z = 0
    std::vector<float> transients;      // [N*N, M]
};

/// Build the derived views (relays, transients) from `rect`. Shared by the
/// real loader and the synthetic replica so both construct the object the
/// same way. `rect` must already be downsampled and gated.
inline void finalize_external(ExternalVolume& v) {
    const int n = v.n_grid, m = v.bins;
    v.relays.assign(size_t(n) * n, RelayPoint{});
    for (int j = 0; j < n; ++j)          // j = y (outer), matching nlos.hpp
        for (int i = 0; i < n; ++i) {    // i = x (inner)
            const float u = (n == 1) ? 0.f : 2.0f * i / (n - 1) - 1.0f;
            const float w = (n == 1) ? 0.f : 2.0f * j / (n - 1) - 1.0f;
            RelayPoint& p = v.relays[size_t(j) * n + i];
            p.position = make_float3(u * v.width, w * v.width, 0.0f);
            p.normal   = make_float3(0.f, 0.f, 1.f);   // into the hidden half-space
        }
    v.transients.assign(size_t(n) * n * m, 0.f);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x)
            for (int t = 0; t < m; ++t)
                v.transients[(size_t(y) * n + x) * m + t] =
                    v.rect[(size_t(x) * n + y) * m + t];
}

/// Downsample the time axis by 2^K by pairwise summation and gate the first
/// z_trim bins -- the exact preprocessing of the released cnlos_reconstruction.m,
/// applied identically to real and replica data.
inline void downsample_and_gate(ExternalVolume& v, int k_down, int z_trim_native,
                                int z_offset_native) {
    int m = v.bins;
    float bin = v.bin_seconds;
    int zt = z_trim_native, zo = z_offset_native;
    for (int k = 0; k < k_down; ++k) {
        const int half = m / 2;
        std::vector<float> out(size_t(v.n_grid) * v.n_grid * half, 0.f);
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = v.rect[2 * i] + v.rect[2 * i + 1];
        // Their loop halves z_trim with round() at each step, not once at the end;
        // for the released values the two agree, but the port keeps their order.
        zt = (zt + 1) / 2;
        zo = (zo + 1) / 2;
        v.rect.swap(out);
        m = half;
        bin *= 2.f;
    }
    for (int x = 0; x < v.n_grid; ++x)
        for (int y = 0; y < v.n_grid; ++y)
            for (int t = 0; t < zt && t < m; ++t)
                v.rect[(size_t(x) * v.n_grid + y) * m + t] = 0.f;
    v.bins = m;
    v.bin_seconds = bin;
    v.z_trim = zt;
    v.z_offset_display = zo;
}

/// Load a converted scene. `base` is the path without extension; both
/// `<base>.bin` and `<base>.meta` must exist.
inline bool load_external(const std::string& base, ExternalVolume& out,
                          std::string& err) {
    std::ifstream meta_file(base + ".meta");
    if (!meta_file) { err = "missing " + base + ".meta"; return false; }

    std::string scene, source;
    int n = 0, bins_native = 0, z_trim = 600, z_off = 0, k_down = 2, isdiffuse = 0;
    float bin_ps = 4.f, width = 0.35f, snr = 0.8f;
    auto trim = [](const std::string& s) {
        const auto a = s.find_first_not_of(" \t");
        const auto b = s.find_last_not_of(" \t\r\n");
        return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
    };
    std::string line;
    while (std::getline(meta_file, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        // BOTH sides are trimmed. Trimming only the value is the bug that
        // costs an afternoon: "scene = x" substr(0, eq) is "scene " with a
        // trailing space, which matches no key and parses as an empty file.
        const std::string key = trim(line.substr(0, eq));
        const std::string v   = trim(line.substr(eq + 1));
        if      (key == "scene")             scene = v;
        else if (key == "source_file")       source = v;
        else if (key == "n_grid")            n = std::stoi(v);
        else if (key == "bins_native")       bins_native = std::stoi(v);
        else if (key == "bin_ps_native")     bin_ps = std::stof(v);
        else if (key == "width_m")           width = std::stof(v);
        else if (key == "z_trim_native")     z_trim = std::stoi(v);
        else if (key == "z_offset_native")   z_off = std::stoi(v);
        else if (key == "isdiffuse")         isdiffuse = std::stoi(v);
        else if (key == "snr")               snr = std::stof(v);
        else if (key == "downsample_k")      k_down = std::stoi(v);
    }
    if (n <= 0 || bins_native <= 0) {
        err = "malformed meta (parsed n=" + std::to_string(n)
            + ", bins=" + std::to_string(bins_native) + ")";
        return false;
    }

    std::ifstream bin_file(base + ".bin", std::ios::binary);
    if (!bin_file) { err = "missing " + base + ".bin"; return false; }
    out = ExternalVolume{};
    out.n_grid = n;
    out.bins = bins_native;
    out.bin_seconds = bin_ps * 1e-12f;
    out.width = width;
    out.diffuse = isdiffuse != 0;
    out.snr = snr;
    out.rect.resize(size_t(n) * n * bins_native);
    bin_file.read(reinterpret_cast<char*>(out.rect.data()),
                  std::streamsize(out.rect.size() * sizeof(float)));
    if (!bin_file) { err = "truncated bin"; return false; }

    downsample_and_gate(out, k_down, z_trim, z_off);
    finalize_external(out);
    std::printf("  loaded %s: %dx%d grid, %d bins @ %.0f ps, wall half-width %.2f m\n",
                scene.c_str(), n, n, out.bins, out.bin_seconds * 1e12f, out.width);
    return true;
}

}  // namespace argos
