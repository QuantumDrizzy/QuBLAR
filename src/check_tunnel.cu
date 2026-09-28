// =============================================================================
// QuBLAR -- ADR-012 part A: confocal NLOS around a mine-drift corner, through dust
// =============================================================================
// Reuses the repo transport (transient_trace's sampling and per-path weight, the CUDA BVH
// traverse_bvh, the splat_pulse time convention) and the repo reconstruction semantics
// (nlos.hpp laplacian_filter + backproject, ported to the GPU). Adds, as ADR-012 sec. 2/4:
//   * relay-spot factor A_p/pi and rho_w^2 so everything is in detected photons relative to
//     B = first-bounce photons per relay point for rho = 1 in clear air;
//   * ballistic dust extinction on the hidden leg (sensor leg applied per relay afterwards);
//   * single-scatter dust backscatter on the hidden leg via a hit-distance histogram;
//   * one waveform per dust level in one pass.
// Usage: check_tunnel <outdir> [log2_paths=20] [seeds=8] [relay_chunk=256] [budget_mult=1]
// =============================================================================
#include "transient.cuh"
#include "nlos.hpp"

#include <curand_kernel.h>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include <fstream>

namespace tun {

constexpr int kNL = 7;                  // dust levels
constexpr int kBins = 1280;
constexpr float kBinS = 100e-12f;
constexpr float kFwhm = 500e-12f;
constexpr int kNR = 32;                 // relay grid side
constexpr int kRelays = kNR * kNR;
constexpr float kSpotArea = 7.853982e-5f;   // 1 cm diameter
constexpr float kOmega0 = 0.9f, kG = 0.7f;
constexpr float kR0 = 0.1f;             // backscatter integration start (m)
constexpr float kTmax = 20.0f;
const float kVis[kNL] = {INFINITY, 50.f, 30.f, 20.f, 15.f, 10.f, 5.f};
double kBudgets[4] = {1e10, 1e11, 1e12, 1e13};   // argv[5] = exploratory multiplier (default 1 = frozen grid)
constexpr double kBgPerB = 1e-14;
// volume
constexpr int VX = 100, VY = 50, VZ = 50;
constexpr float kVloX = 4.f, kVloY = 0.f, kVloZ = 1.f, kVhiX = 14.f, kVhiY = 5.f, kVhiZ = 6.f;
const float3 kVlo = {kVloX, kVloY, kVloZ}, kVhi = {kVhiX, kVhiY, kVhiZ};

struct DustParams { float sig_t[kNL]; float bs_coef[kNL]; };   // bs_coef = sigma_s * p(pi)

struct TunRelay { float3 pos; float3 nrm; float rho; float L1; };

struct TunTarget { int kind; float cx, cz; float lo[3], hi[3]; };

static void add_box(Scene& s, float x0, float x1, float y0, float y1, float z0, float z1, int m) {
    s.add_quad({x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1}, m);
    s.add_quad({x0, y1, z0}, {x1, y1, z0}, {x1, y0, z0}, {x0, y0, z0}, m);
    s.add_quad({x1, y0, z0}, {x1, y1, z0}, {x1, y1, z1}, {x1, y0, z1}, m);
    s.add_quad({x0, y0, z1}, {x0, y1, z1}, {x0, y1, z0}, {x0, y0, z0}, m);
    s.add_quad({x0, y1, z0}, {x0, y1, z1}, {x1, y1, z1}, {x1, y1, z0}, m);
    s.add_quad({x0, y0, z1}, {x0, y0, z0}, {x1, y0, z0}, {x1, y0, z1}, m);
}

// kind 0 = empty, 1 = person, 2 = vehicle
static Scene make_tunnel(int kind, float cx, float cz, TunTarget& tgt) {
    Scene s;
    s.add_material(0.20f);   // 0 rock
    s.add_material(0.50f);   // 1 person
    s.add_material(0.30f);   // 2 vehicle
    // drift A
    s.add_quad({-2.5f, 0, -10}, {2.5f, 0, -10}, {2.5f, 0, 6}, {-2.5f, 0, 6}, 0);      // floor
    s.add_quad({-2.5f, 5, -10}, {-2.5f, 5, 6}, {2.5f, 5, 6}, {2.5f, 5, -10}, 0);      // ceiling
    s.add_quad({-2.5f, 0, -10}, {-2.5f, 0, 6}, {-2.5f, 5, 6}, {-2.5f, 5, -10}, 0);    // left
    s.add_quad({2.5f, 0, -10}, {2.5f, 5, -10}, {2.5f, 5, 1}, {2.5f, 0, 1}, 0);        // right (to corner)
    s.add_quad({-2.5f, 0, -10}, {-2.5f, 5, -10}, {2.5f, 5, -10}, {2.5f, 0, -10}, 0);  // back
    s.add_quad({-2.5f, 0, 6}, {22.5f, 0, 6}, {22.5f, 5, 6}, {-2.5f, 5, 6}, 0);        // end wall z=6
    // drift B
    s.add_quad({2.5f, 0, 1}, {22.5f, 0, 1}, {22.5f, 0, 6}, {2.5f, 0, 6}, 0);          // floor
    s.add_quad({2.5f, 5, 1}, {2.5f, 5, 6}, {22.5f, 5, 6}, {22.5f, 5, 1}, 0);          // ceiling
    s.add_quad({2.5f, 0, 1}, {2.5f, 5, 1}, {22.5f, 5, 1}, {22.5f, 0, 1}, 0);          // wall z=1
    s.add_quad({22.5f, 0, 1}, {22.5f, 5, 1}, {22.5f, 5, 6}, {22.5f, 0, 6}, 0);        // end x=22.5
    tgt.kind = kind; tgt.cx = cx; tgt.cz = cz;
    if (kind == 1) {
        add_box(s, cx - 0.175f, cx - 0.025f, 0.0f, 0.85f, cz - 0.10f, cz + 0.10f, 1);
        add_box(s, cx + 0.025f, cx + 0.175f, 0.0f, 0.85f, cz - 0.10f, cz + 0.10f, 1);
        add_box(s, cx - 0.225f, cx + 0.225f, 0.85f, 1.45f, cz - 0.15f, cz + 0.15f, 1);
        add_box(s, cx - 0.10f, cx + 0.10f, 1.45f, 1.70f, cz - 0.10f, cz + 0.10f, 1);
        tgt.lo[0] = cx - 0.225f; tgt.hi[0] = cx + 0.225f; tgt.lo[1] = 0; tgt.hi[1] = 1.7f;
        tgt.lo[2] = cz - 0.15f; tgt.hi[2] = cz + 0.15f;
    } else if (kind == 2) {
        add_box(s, cx - 1.25f, cx + 1.25f, 0.3f, 1.3f, cz - 0.75f, cz + 0.75f, 2);
        add_box(s, cx - 1.25f, cx - 0.25f, 1.3f, 1.8f, cz - 0.75f, cz + 0.75f, 2);
        tgt.lo[0] = cx - 1.25f; tgt.hi[0] = cx + 1.25f; tgt.lo[1] = 0.3f; tgt.hi[1] = 1.8f;
        tgt.lo[2] = cz - 0.75f; tgt.hi[2] = cz + 0.75f;
    }
    s.build();
    return s;
}

// ---------------------------------------------------------------------------
// Transport kernel: one block per relay point (copy of transient_trace, extended)
// ---------------------------------------------------------------------------
__global__ void tunnel_trace(const BvhNode* __restrict__ nodes, const int* __restrict__ indices,
                             const Triangle* __restrict__ tris, const Material* __restrict__ mats,
                             const TunRelay* __restrict__ relays, int relay_offset, int n_chunk,
                             DustParams dust, float t0, int paths, unsigned seed,
                             float* __restrict__ out,        // [kNL][kRelays][kBins], no sensor-leg attenuation
                             float* __restrict__ dust_sum)   // [kNL][kRelays] dust-backscatter share
{
    const int rp = relay_offset + blockIdx.x;
    if (blockIdx.x >= n_chunk) return;
    extern __shared__ float sm[];
    float* wave = sm;                         // kNL * kBins
    float* hist = sm + kNL * kBins;           // kBins (range histogram of sum cos_p)
    for (int k = threadIdx.x; k < (kNL + 1) * kBins; k += blockDim.x) sm[k] = 0.f;
    __syncthreads();

    const TunRelay R = relays[rp];
    const float c = static_cast<float>(kSpeedOfLight);
    const float dr = 0.5f * c * kBinS;
    const float inv_bin = 1.0f / kBinS;

    for (int i = threadIdx.x; i < paths; i += blockDim.x) {
        unsigned key = hash_u32(static_cast<unsigned>(rp) * 0x9E3779B9u
                              ^ static_cast<unsigned>(i) * 0x85EBCA6Bu ^ seed);
        const float u1 = (hash_u32(key) >> 8) * (1.0f / 16777216.0f);
        const float u2 = (hash_u32(key ^ 0x68BC21EBu) >> 8) * (1.0f / 16777216.0f);
        const float3 dir = argos::cosine_hemisphere(R.nrm, u1, u2);
        if (dir.z > -0.01f) continue;                 // into the (flat) wall: lost
        const float cos_p = fabsf(d_dot(R.nrm, dir));
        Ray ray;
        ray.origin = R.pos + R.nrm * 1e-4f;
        ray.direction = dir;
        ray.tmax = kTmax;
        const Hit h = traverse_bvh(nodes, indices, tris, ray);
        int hb = kBins - 1;
        if (h.t > 0.0f) {
            const float rr = h.t;
            hb = min(kBins - 1, static_cast<int>(rr / dr));
            const float cos_q = fabsf(d_dot(h.normal, dir));
            const float rho_q = mats[h.material].reflectance;
            const float w = rho_q * cos_q * cos_p / (rr * rr);
            const float t = 2.0f * (R.L1 + rr) / c;
            const float fk = (t - t0) * inv_bin - 0.5f;
            const int k = static_cast<int>(floorf(fk));
            if (k >= 0 && k + 1 < kBins) {
                const float f = fk - k;
                #pragma unroll
                for (int l = 0; l < kNL; ++l) {
                    const float a = w * __expf(-2.0f * dust.sig_t[l] * rr);
                    atomicAdd(&wave[l * kBins + k], a * (1.f - f));
                    atomicAdd(&wave[l * kBins + k + 1], a * f);
                }
            }
        }
        atomicAdd(&hist[hb], cos_p);
    }
    __syncthreads();

    // surface part scale: rho_w^2 * A_p/pi / paths ; dust part: rho_w^2 * A_p * sigma_s p(pi) / paths
    const float inv_paths = 1.0f / paths;
    const float s_surf = R.rho * R.rho * (kSpotArea / 3.14159265f) * inv_paths;
    const float s_dust = R.rho * R.rho * kSpotArea * inv_paths;
    if (threadIdx.x < kNL) {
        const int l = threadIdx.x;
        // scale this level's surface wave first (only this thread touches level l here)
        for (int k = 0; k < kBins; ++k) wave[l * kBins + k] *= s_surf;
        double tot = 0.0;
        if (dust.bs_coef[l] > 0.f) {
            float cum = 0.f;          // sum of cos_p for paths with r_hit beyond bin j
            for (int j = kBins - 1; j >= 0; --j) {
                const float Hj = cum + 0.5f * hist[j];
                cum += hist[j];
                const float r = (j + 0.5f) * dr;
                if (r < kR0) continue;
                const float a = s_dust * dust.bs_coef[l] * Hj * __expf(-2.0f * dust.sig_t[l] * r) / (r * r) * dr;
                const float t = 2.0f * (R.L1 + r) / c;
                const float fk = (t - t0) * inv_bin - 0.5f;
                const int k = static_cast<int>(floorf(fk));
                if (k >= 0 && k + 1 < kBins) {
                    const float f = fk - k;
                    wave[l * kBins + k] += a * (1.f - f);
                    wave[l * kBins + k + 1] += a * f;
                    tot += a;
                }
            }
        }
        dust_sum[l * kRelays + rp] = static_cast<float>(tot);
    }
    __syncthreads();
    // pulse: bin-integrated Gaussian about each bin centre (same integral as splat_pulse)
    const float sig = kFwhm / 2.354820045f;
    float g[19];
    const float inv = 1.0f / (sig * 1.41421356f);
    #pragma unroll
    for (int j = -9; j <= 9; ++j)
        g[j + 9] = 0.5f * (erff(((j + 0.5f) * kBinS) * inv) - erff(((j - 0.5f) * kBinS) * inv));
    for (int idx = threadIdx.x; idx < kNL * kBins; idx += blockDim.x) {
        const int l = idx / kBins, k = idx % kBins;
        float acc = 0.f;
        #pragma unroll
        for (int j = -9; j <= 9; ++j) {
            const int kk = k - j;
            if (kk >= 0 && kk < kBins) acc += g[j + 9] * wave[l * kBins + kk];
        }
        out[(static_cast<size_t>(l) * kRelays + rp) * kBins + k] = acc;
    }
}

// ---------------------------------------------------------------------------
// Noise + change detection: counts = Pois(B*att*E + bg) [- Pois(B*att*E_ref + bg)]
// ---------------------------------------------------------------------------
__global__ void make_counts(const float* __restrict__ E, const float* __restrict__ Eref,
                            const float* __restrict__ att, double B, double bg, int noisy,
                            unsigned long long key, float* __restrict__ counts)
{
    const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= static_cast<size_t>(kRelays) * kBins) return;
    const int r = static_cast<int>(idx / kBins);
    const double lam = B * att[r] * E[idx] + bg;
    double v;
    if (!noisy) {
        v = lam - (Eref ? B * att[r] * Eref[idx] + bg : 0.0);
    } else {
        curandStatePhilox4_32_10_t st;
        curand_init(key, idx, 0, &st);
        v = static_cast<double>(curand_poisson(&st, lam));
        if (Eref) {
            const double lr = B * att[r] * Eref[idx] + bg;
            curandStatePhilox4_32_10_t st2;
            curand_init(key ^ 0x9E3779B97F4A7C15ull, idx, 0, &st2);
            v -= static_cast<double>(curand_poisson(&st2, lr));
        }
    }
    counts[idx] = static_cast<float>(v);
}

__global__ void laplace_t(const float* __restrict__ in, float* __restrict__ out) {
    const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= static_cast<size_t>(kRelays) * kBins) return;
    const int k = static_cast<int>(idx % kBins);
    out[idx] = (k == 0 || k == kBins - 1) ? 0.f : (-in[idx - 1] + 2.f * in[idx] - in[idx + 1]);
}

// Expected per-bin variance of the Laplacian-filtered counts (Poisson, independent bins;
// P2 adds the reference scan's variance). Exploratory whitening only (not a frozen rule).
__global__ void make_var(const float* __restrict__ E, const float* __restrict__ Eref,
                         const float* __restrict__ att, double B, double bg, float* __restrict__ var)
{
    const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= static_cast<size_t>(kRelays) * kBins) return;
    const int r = static_cast<int>(idx / kBins);
    const int k = static_cast<int>(idx % kBins);
    if (k == 0 || k == kBins - 1) { var[idx] = 0.f; return; }
    auto lam = [&](size_t i) {
        double v = B * att[r] * E[i] + bg;
        if (Eref) v += B * att[r] * Eref[i] + bg;
        return v;
    };
    var[idx] = static_cast<float>(lam(idx - 1) + 4.0 * lam(idx) + lam(idx + 1));
}

__global__ void backproject_var_gpu(const float* __restrict__ var, const float4* __restrict__ relay_pl1,
                                    float t0, float* __restrict__ vol)
{
    __shared__ float4 rs[kRelays];
    for (int i = threadIdx.x; i < kRelays; i += blockDim.x) rs[i] = relay_pl1[i];
    __syncthreads();
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= VX * VY * VZ) return;
    const int ix = v % VX, iy = (v / VX) % VY, iz = v / (VX * VY);
    const float x = kVloX + (kVhiX - kVloX) * (ix + 0.5f) / VX;
    const float y = kVloY + (kVhiY - kVloY) * (iy + 0.5f) / VY;
    const float z = kVloZ + (kVhiZ - kVloZ) * (iz + 0.5f) / VZ;
    const float c = static_cast<float>(kSpeedOfLight);
    float acc = 0.f;
    for (int r = 0; r < kRelays; ++r) {
        const float4 p = rs[r];
        const float dx = x - p.x, dy = y - p.y, dz = z - p.z;
        const float rr = sqrtf(dx * dx + dy * dy + dz * dz);
        const float t = 2.0f * (p.w + rr) / c;
        const float fk = (t - t0) / kBinS - 0.5f;
        const int k = static_cast<int>(floorf(fk));
        if (k < 0 || k + 1 >= kBins) continue;
        const float f = fk - k;
        const float* w = var + static_cast<size_t>(r) * kBins;
        acc += (1.f - f) * (1.f - f) * w[k] + f * f * w[k + 1];
    }
    vol[v] = acc;
}

// GPU port of argos::backproject (same bin interpolation and time convention)
__global__ void backproject_gpu(const float* __restrict__ tr, const float4* __restrict__ relay_pl1,
                                float t0, float* __restrict__ vol)
{
    __shared__ float4 rs[kRelays];
    for (int i = threadIdx.x; i < kRelays; i += blockDim.x) rs[i] = relay_pl1[i];
    __syncthreads();
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= VX * VY * VZ) return;
    const int ix = v % VX, iy = (v / VX) % VY, iz = v / (VX * VY);
    const float x = kVloX + (kVhiX - kVloX) * (ix + 0.5f) / VX;
    const float y = kVloY + (kVhiY - kVloY) * (iy + 0.5f) / VY;
    const float z = kVloZ + (kVhiZ - kVloZ) * (iz + 0.5f) / VZ;
    const float c = static_cast<float>(kSpeedOfLight);
    float acc = 0.f;
    for (int r = 0; r < kRelays; ++r) {
        const float4 p = rs[r];
        const float dx = x - p.x, dy = y - p.y, dz = z - p.z;
        const float rr = sqrtf(dx * dx + dy * dy + dz * dz);
        const float t = 2.0f * (p.w + rr) / c;
        const float fk = (t - t0) / kBinS - 0.5f;
        const int k = static_cast<int>(floorf(fk));
        if (k < 0 || k + 1 >= kBins) continue;
        const float f = fk - k;
        const float* w = tr + static_cast<size_t>(r) * kBins;
        acc += (1.f - f) * w[k] + f * w[k + 1];
    }
    vol[v] = acc;
}

}  // namespace tun

using namespace tun;

static inline float dist_to_aabb(float x, float y, float z, const TunTarget& t) {
    const float dx = std::max({t.lo[0] - x, 0.f, x - t.hi[0]});
    const float dy = std::max({t.lo[1] - y, 0.f, y - t.hi[1]});
    const float dz = std::max({t.lo[2] - z, 0.f, z - t.hi[2]});
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

int main(int argc, char** argv) {
    const std::string out = argc > 1 ? argv[1] : "build/tunnel";
    const int log2p = argc > 2 ? std::atoi(argv[2]) : 20;
    const int nseeds = argc > 3 ? std::atoi(argv[3]) : 8;
    const int chunk = argc > 4 ? std::atoi(argv[4]) : 256;
    const int paths = 1 << log2p;
    const double bmult = argc > 5 ? std::atof(argv[5]) : 1.0;
    for (int b = 0; b < 4; ++b) kBudgets[b] *= bmult;
    const auto T0 = std::chrono::steady_clock::now();
    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count(); };

    DustParams dust;
    const float pbk = (1.f - kG) / (4.f * 3.14159265f * (1.f + kG) * (1.f + kG));
    for (int l = 0; l < kNL; ++l) {
        dust.sig_t[l] = std::isinf(kVis[l]) ? 0.f : 3.912f / kVis[l];
        dust.bs_coef[l] = kOmega0 * dust.sig_t[l] * pbk;
    }
    const float3 s_far = {-1.5f, 1.6f, -4.f}, s_near = {0.f, 1.6f, 3.f};
    std::vector<float3> rpos(kRelays);
    std::vector<float> L1f(kRelays), L1n(kRelays);
    float L1min = 1e9f;
    for (int j = 0; j < kNR; ++j) for (int i = 0; i < kNR; ++i) {
        const int r = j * kNR + i;
        rpos[r] = make_float3(1.f + 4.f * (i + 0.5f) / kNR, 0.5f + 4.f * (j + 0.5f) / kNR, 6.f);
        auto d = [&](float3 s) { float dx = rpos[r].x - s.x, dy = rpos[r].y - s.y, dz = rpos[r].z - s.z;
                                 return std::sqrt(dx * dx + dy * dy + dz * dz); };
        L1f[r] = d(s_far); L1n[r] = d(s_near); L1min = std::min(L1min, L1f[r]);
    }
    const float t0 = 2.f * L1min / static_cast<float>(kSpeedOfLight) - 1e-9f;
    {
        argos::Voxels vx; vx.lo = kVlo; vx.hi = kVhi; vx.nx = VX; vx.ny = VY; vx.nz = VZ;
        argos::SensorConfig sc; sc.bins = kBins; sc.bin_seconds = kBinS; sc.pulse_fwhm_seconds = kFwhm; sc.t0_seconds = t0;
        std::printf("grid_to_pulse_ratio = %.3f  t0 = %.3f ns  L1 far [%.2f..]  paths/relay = %d\n",
                    argos::grid_to_pulse_ratio(vx, sc), t0 * 1e9, L1min, paths);
    }

    // interior mask
    std::vector<unsigned char> interior(VX * VY * VZ, 0);
    std::vector<float3> vc(VX * VY * VZ);
    for (int iz = 0; iz < VZ; ++iz) for (int iy = 0; iy < VY; ++iy) for (int ix = 0; ix < VX; ++ix) {
        const int v = (iz * VY + iy) * VX + ix;
        const float x = kVlo.x + (kVhi.x - kVlo.x) * (ix + 0.5f) / VX;
        const float y = kVlo.y + (kVhi.y - kVlo.y) * (iy + 0.5f) / VY;
        const float z = kVlo.z + (kVhi.z - kVlo.z) * (iz + 0.5f) / VZ;
        vc[v] = make_float3(x, y, z);
        interior[v] = (y >= 0.3f && y <= 4.7f && z >= 1.3f && z <= 5.7f && x >= 4.3f && x <= 13.7f);
    }

    const size_t NE = static_cast<size_t>(kNL) * kRelays * kBins;
    float* dE[3]; float* dDust[3];
    for (int s = 0; s < 3; ++s) { CUDA_CHECK(cudaMalloc(&dE[s], NE * sizeof(float))); CUDA_CHECK(cudaMalloc(&dDust[s], kNL * kRelays * sizeof(float))); }
    float *dCounts, *dFilt, *dVol, *dAtt, *dVar, *dVolVar; float4* dRelPL; TunRelay* dRel;
    CUDA_CHECK(cudaMalloc(&dCounts, static_cast<size_t>(kRelays) * kBins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dFilt, static_cast<size_t>(kRelays) * kBins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dVol, VX * VY * VZ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dVolVar, VX * VY * VZ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dVar, static_cast<size_t>(kRelays) * kBins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dAtt, 2 * kNL * kRelays * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRelPL, kRelays * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dRel, kRelays * sizeof(TunRelay)));
    {
        std::vector<float4> pl(kRelays);
        for (int r = 0; r < kRelays; ++r) pl[r] = make_float4(rpos[r].x, rpos[r].y, rpos[r].z, L1f[r]);
        CUDA_CHECK(cudaMemcpy(dRelPL, pl.data(), kRelays * sizeof(float4), cudaMemcpyHostToDevice));
        std::vector<float> att(2 * kNL * kRelays);
        for (int sn = 0; sn < 2; ++sn) for (int l = 0; l < kNL; ++l) for (int r = 0; r < kRelays; ++r)
            att[(sn * kNL + l) * kRelays + r] = std::exp(-2.f * dust.sig_t[l] * (sn == 0 ? L1f[r] : L1n[r]));
        CUDA_CHECK(cudaMemcpy(dAtt, att.data(), att.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    const size_t smem = (kNL + 1) * kBins * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(tunnel_trace, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    FILE* ftr = std::fopen((out + "/tunnel_trials.csv").c_str(), "w");
    std::fprintf(ftr, "seed,sensor,V,B,pipe,scene,T,px,py,pz,dist,Tw,wx,wy,wz,wdist\n");
    FILE* fph = std::fopen((out + "/tunnel_photons.csv").c_str(), "w");
    std::fprintf(fph, "seed,sensor,V,B,scene,total_photons,dust_photons,net_vs_empty\n");
    FILE* fmeta = std::fopen((out + "/tunnel_targets.csv").c_str(), "w");
    std::fprintf(fmeta, "seed,scene,cx,cz,lo_x,lo_y,lo_z,hi_x,hi_y,hi_z\n");
    FILE* fmip = std::fopen((out + "/tunnel_mips.f32").c_str(), "wb");
    FILE* fmipi = std::fopen((out + "/tunnel_mips_index.csv").c_str(), "w");
    std::fprintf(fmipi, "i,seed,sensor,V,B,pipe,scene,noisy\n");
    int mip_i = 0;
    const char* scn[3] = {"empty", "person", "vehicle"};
    const char* sen[2] = {"far", "near"};
    double t_render = 0, t_recon = 0;

    std::vector<float> vol(VX * VY * VZ), volvar(VX * VY * VZ);
    auto run_volume = [&](int si, int sn, int l, double B, int pipe, int noisy, unsigned long long key) {
        const float* Eref = (pipe == 2) ? (dE[0] + static_cast<size_t>(l) * kRelays * kBins) : nullptr;
        const size_t n = static_cast<size_t>(kRelays) * kBins;
        make_counts<<<(unsigned)((n + 255) / 256), 256>>>(dE[si] + static_cast<size_t>(l) * kRelays * kBins, Eref,
                                              dAtt + (sn * kNL + l) * kRelays, B, kBgPerB * B, noisy, key, dCounts);
        laplace_t<<<(unsigned)((n + 255) / 256), 256>>>(dCounts, dFilt);
        backproject_gpu<<<(VX * VY * VZ + 255) / 256, 256>>>(dFilt, dRelPL, t0, dVol);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(vol.data(), dVol, vol.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (noisy) {
            make_var<<<(unsigned)((n + 255) / 256), 256>>>(dE[si] + static_cast<size_t>(l) * kRelays * kBins, Eref,
                                                  dAtt + (sn * kNL + l) * kRelays, B, kBgPerB * B, dVar);
            backproject_var_gpu<<<(VX * VY * VZ + 255) / 256, 256>>>(dVar, dRelPL, t0, dVolVar);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(volvar.data(), dVolVar, volvar.size() * sizeof(float), cudaMemcpyDeviceToHost));
        }
    };
    auto save_mip = [&](int seed, int sn, int l, double B, int pipe, int si, int noisy) {
        std::vector<float> top(VX * VZ, -1e30f), side(VX * VY, -1e30f);
        for (int iz = 0; iz < VZ; ++iz) for (int iy = 0; iy < VY; ++iy) for (int ix = 0; ix < VX; ++ix) {
            const float a = vol[(iz * VY + iy) * VX + ix];
            top[iz * VX + ix] = std::max(top[iz * VX + ix], a);
            side[iy * VX + ix] = std::max(side[iy * VX + ix], a);
        }
        std::fwrite(top.data(), sizeof(float), top.size(), fmip);
        std::fwrite(side.data(), sizeof(float), side.size(), fmip);
        std::fprintf(fmipi, "%d,%d,%s,%g,%g,P%d,%s,%d\n", mip_i++, seed, sen[sn], kVis[l], B, pipe, scn[si], noisy);
    };

    for (int seed = 1; seed <= nseeds; ++seed) {
        std::mt19937_64 rng(0xADC012ull * 1000003ull + seed);
        std::uniform_real_distribution<float> U(0.f, 1.f);
        std::vector<TunRelay> rel(kRelays);
        for (int r = 0; r < kRelays; ++r) {
            const float th = U(rng) * 5.f * 3.14159265f / 180.f, ph = U(rng) * 6.2831853f;
            rel[r].pos = rpos[r];
            rel[r].nrm = make_float3(std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), -std::cos(th));
            rel[r].rho = 0.10f + 0.20f * U(rng);
            rel[r].L1 = L1f[r];
        }
        CUDA_CHECK(cudaMemcpy(dRel, rel.data(), kRelays * sizeof(TunRelay), cudaMemcpyHostToDevice));
        const float jx1 = U(rng) - 0.5f, jz1 = U(rng) - 0.5f, jx2 = U(rng) - 0.5f, jz2 = U(rng) - 0.5f;
        TunTarget tg[3];
        const double tr0 = secs();
        for (int si = 0; si < 3; ++si) {
            const float cx = si == 1 ? 8.f + jx1 : (si == 2 ? 10.f + jx2 : 0.f);
            const float cz = si == 1 ? 3.5f + jz1 : (si == 2 ? 3.5f + jz2 : 0.f);
            Scene sc = make_tunnel(si, cx, cz, tg[si]);
            DeviceScene ds; ds.upload(sc);
            for (int off = 0; off < kRelays; off += chunk) {
                const int nc = std::min(chunk, kRelays - off);
                tunnel_trace<<<nc, 256, smem>>>(ds.nodes, ds.indices, ds.tris, ds.mats, dRel, off, nc, dust, t0,
                                                paths, 0x51ED270Bu ^ (seed * 0x01000193u), dE[si], dDust[si]);   // common random numbers across scenes
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            ds.free();
            if (si > 0) std::fprintf(fmeta, "%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", seed, scn[si], cx, cz,
                                     tg[si].lo[0], tg[si].lo[1], tg[si].lo[2], tg[si].hi[0], tg[si].hi[1], tg[si].hi[2]);
        }
        t_render += secs() - tr0;
        // photon accounting (expected, per scan)
        {
            std::vector<float> hE(NE), hD(kNL * kRelays);
            std::vector<double> tot[3], dus[3];
            for (int si = 0; si < 3; ++si) {
                CUDA_CHECK(cudaMemcpy(hE.data(), dE[si], NE * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hD.data(), dDust[si], hD.size() * sizeof(float), cudaMemcpyDeviceToHost));
                tot[si].assign(2 * kNL, 0.0); dus[si].assign(2 * kNL, 0.0);
                for (int sn = 0; sn < 2; ++sn) for (int l = 0; l < kNL; ++l) for (int r = 0; r < kRelays; ++r) {
                    const double a = std::exp(-2.0 * dust.sig_t[l] * (sn == 0 ? L1f[r] : L1n[r]));
                    double s = 0; const float* w = &hE[(static_cast<size_t>(l) * kRelays + r) * kBins];
                    for (int k = 0; k < kBins; ++k) s += w[k];
                    tot[si][sn * kNL + l] += a * s; dus[si][sn * kNL + l] += a * hD[l * kRelays + r];
                }
                if (seed == 1 && si > 0) {   // example waveforms at the patch centre relay, far sensor
                    FILE* fw = std::fopen((out + "/tunnel_wave_" + scn[si] + ".f32").c_str(), "wb");
                    const int rc = (kNR / 2) * kNR + kNR / 2;
                    for (int l = 0; l < kNL; ++l) {
                        std::vector<float> w(kBins);
                        const float a = std::exp(-2.f * dust.sig_t[l] * L1f[rc]);
                        for (int k = 0; k < kBins; ++k) w[k] = a * hE[(static_cast<size_t>(l) * kRelays + rc) * kBins + k];
                        std::fwrite(w.data(), sizeof(float), kBins, fw);
                    }
                    std::fclose(fw);
                }
                if (seed == 1 && si == 0) {
                    FILE* fw = std::fopen((out + "/tunnel_wave_empty.f32").c_str(), "wb");
                    const int rc = (kNR / 2) * kNR + kNR / 2;
                    for (int l = 0; l < kNL; ++l) {
                        std::vector<float> w(kBins);
                        const float a = std::exp(-2.f * dust.sig_t[l] * L1f[rc]);
                        for (int k = 0; k < kBins; ++k) w[k] = a * hE[(static_cast<size_t>(l) * kRelays + rc) * kBins + k];
                        std::fwrite(w.data(), sizeof(float), kBins, fw);
                    }
                    std::fclose(fw);
                }
            }
            for (int sn = 0; sn < 2; ++sn) for (int l = 0; l < kNL; ++l) for (int b = 0; b < 4; ++b) for (int si = 0; si < 3; ++si)
                std::fprintf(fph, "%d,%s,%g,%g,%s,%.6e,%.6e,%.6e\n", seed, sen[sn], kVis[l], kBudgets[b], scn[si],
                             kBudgets[b] * tot[si][sn * kNL + l], kBudgets[b] * dus[si][sn * kNL + l],
                             kBudgets[b] * (tot[si][sn * kNL + l] - tot[0][sn * kNL + l]));
        }
        const double tc0 = secs();
        for (int sn = 0; sn < 2; ++sn) for (int l = 0; l < kNL; ++l) for (int b = 0; b < 4; ++b)
        for (int pipe = 1; pipe <= 2; ++pipe) for (int si = 0; si < 3; ++si) {
            const unsigned long long key = (static_cast<unsigned long long>(seed) << 40) ^ (static_cast<unsigned long long>(sn) << 32)
                ^ (static_cast<unsigned long long>(l) << 24) ^ (static_cast<unsigned long long>(b) << 16)
                ^ (static_cast<unsigned long long>(pipe) << 8) ^ static_cast<unsigned long long>(si) ^ 0xC0FFEE1234ull;
            run_volume(si, sn, l, kBudgets[b], pipe, 1, key);
            int best = -1; float bv = -1e30f;
            for (int v = 0; v < VX * VY * VZ; ++v) if (interior[v] && vol[v] > bv) { bv = vol[v]; best = v; }
            const float3 p = vc[best];
            const float dist = si > 0 ? dist_to_aabb(p.x, p.y, p.z, tg[si]) : -1.f;
            // exploratory (NOT frozen): noise-whitened map vol / sd, max over interior
            int bw = -1; float bwv = -1e30f;
            for (int v = 0; v < VX * VY * VZ; ++v) if (interior[v] && volvar[v] > 0.f) {
                const float zz = vol[v] / std::sqrt(volvar[v]);
                if (zz > bwv) { bwv = zz; bw = v; }
            }
            const float3 q = vc[bw];
            const float wdist = si > 0 ? dist_to_aabb(q.x, q.y, q.z, tg[si]) : -1.f;
            std::fprintf(ftr, "%d,%s,%g,%g,P%d,%s,%.6e,%.3f,%.3f,%.3f,%.4f,%.5f,%.3f,%.3f,%.3f,%.4f\n", seed, sen[sn], kVis[l], kBudgets[b], pipe, scn[si],
                         bv, p.x, p.y, p.z, dist, bwv, q.x, q.y, q.z, wdist);
            if (seed == 1 && sn == 0 && (kBudgets[b] == 1e12 || b == 3) && (l == 0 || l == 2 || l == 3 || l == 4)) {
                save_mip(seed, sn, l, kBudgets[b], pipe, si, 1);
                for (int v = 0; v < VX * VY * VZ; ++v) vol[v] = volvar[v] > 0.f ? vol[v] / std::sqrt(volvar[v]) : 0.f;
                save_mip(seed, sn, l, kBudgets[b], pipe, si, 2);   // 2 = noise-whitened (exploratory)
            }
        }
        if (seed == 1) {   // noiseless reference volumes (expected counts), far sensor, B = 1e12
            for (int l : {0, 3}) for (int pipe = 1; pipe <= 2; ++pipe) for (int si = 0; si < 3; ++si) {
                run_volume(si, 0, l, 1e12, pipe, 0, 0);
                save_mip(seed, 0, l, 1e12, pipe, si, 0);
                if (l == 0 && pipe == 2 && si > 0) {
                    FILE* fv = std::fopen((out + "/tunnel_vol_" + scn[si] + "_P2_clean.f32").c_str(), "wb");
                    std::fwrite(vol.data(), sizeof(float), vol.size(), fv); std::fclose(fv);
                }
            }
            for (int si = 1; si < 3; ++si) {   // one noisy 3D volume for rendering
                run_volume(si, 0, 0, 1e12, 2, 1, 0xBEEFull + si);
                FILE* fv = std::fopen((out + "/tunnel_vol_" + scn[si] + "_P2_B1e12.f32").c_str(), "wb");
                std::fwrite(vol.data(), sizeof(float), vol.size(), fv); std::fclose(fv);
            }
        }
        t_recon += secs() - tc0;
        std::fflush(ftr); std::fflush(fph);
        std::printf("seed %d done  (render %.1f s, recon %.1f s cumulative, wall %.1f s)\n", seed, t_render, t_recon, secs());
        std::fflush(stdout);
    }
    std::fclose(ftr); std::fclose(fph); std::fclose(fmeta); std::fclose(fmip); std::fclose(fmipi);
    FILE* fm = std::fopen((out + "/tunnel_meta.txt").c_str(), "w");
    std::fprintf(fm, "budget_mult=%g\npaths_per_relay=%d\nseeds=%d\nbins=%d\nbin_ps=%.1f\nt0_ns=%.4f\nrelays=%d\nvox=%dx%dx%d\nrender_s=%.2f\nrecon_s=%.2f\nwall_s=%.2f\n",
                 bmult, paths, nseeds, kBins, kBinS * 1e12, t0 * 1e9, kRelays, VX, VY, VZ, t_render, t_recon, secs());
    std::fclose(fm);
    std::printf("done in %.1f s\n", secs());
    return 0;
}
