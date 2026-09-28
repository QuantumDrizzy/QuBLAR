// =============================================================================
// QuBLAR -- check_khufu: muography VALIDATION against the published Khufu voids
// =============================================================================
// ADR-011 (pre-registered; frozen hash in experiments/khufu/PREREG_SHA256.txt).
// Synthetic only. The muon machinery is COPIED from src/check_ark.cu (ADR-010),
// which is not modified: Reyna/CSDA survival table, hash-keyed expected maps,
// Bernoulli cross-check, profile-likelihood statistic with the Hermite/golden
// profile over the density scale, mlem_transmission for the density figure.
// New here: the pyramid scene (known chambers, Big Void, North Face Corridor,
// a blind void), per-detector acceptance windows and exposures, the template
// scan localisation, the direction check and the paper comparison (ADR-011 s5).
// =============================================================================

#include "muon.cuh"
#include "muon_recon.hpp"
#include "muon_replica.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace argos;

// ---------------------------------------------------------------- ADR-011 s2
static const float kVox = 0.5f;
static const int kNX = 480, kNY = 480, kNZ = 284;
static const float3 kGridLo = make_float3(-120.0f, -120.0f, -2.0f);
static const float kPyrHalfBase = 115.15f, kHorig = 146.6f, kHtop = 138.5f;
static const float kXP = 7.2f;
static const float kRhoAir = 0.0012f, kRhoStone = 2.2f, kRhoGranite = 2.75f;
static const float kTanDC = 0.498581f;   // tan 26.5 deg
static const float kTanGG = 0.494231f;   // tan 26.3 deg

enum SceneId { S_K0 = 0, S_K1, S_K1I, S_K2, S_K12, S_KB, S_N };
static const char* scene_name(int s) {
    static const char* n[] = {"K0", "K1", "K1i", "K2", "K12", "KB"};
    return n[s];
}
static const int kNModel = 5;   // K0 K1 K1i K2 K12

// Big Void (K1), NFC (K2)
static const float3 kBVc = make_float3(kXP, 15.0f, 60.0f);
static const float3 kBVh = make_float3(0.8f, 15.0f, 4.3f);
static const float3 kNFCc = make_float3(kXP, 92.5f, 21.0f);
static const float3 kNFCh = make_float3(1.0f, 4.5f, 1.0f);

// ---------------------------------------------------------------- ADR-011 s4
struct KDet {
    const char* name;
    float3 p;
    double area, days;
    int cone;        // 0: box |tan| <= a ; 1: cone of half-angle a (deg) around dir to kConeAim
    float a;
    int set;         // 0 Big-Void set, 1 NFC set
    double paper_tracks;   // published tracks/events (0 = n/a)
    double paper_Z;        // published significance; negative = lower bound "> |Z|"; 0 = combined elsewhere
};
static float zDC(float y) { return 17.0f - (101.8f - y) * kTanDC; }
static const float3 kConeAim = make_float3(kXP, 18.0f, 47.5f);
static const int kNDet = 13;
static KDet kDet[kNDet] = {
    {"NE1", make_float3(13.0f, 0.55f, 22.0f), 0.45, 98, 0, 1.0f, 0, 4.4e6, 13.7},
    {"NE2", make_float3(2.7f, -0.55f, 22.0f), 0.45, 140, 0, 1.0f, 0, 6.2e6, 12.7},
    {"H1", make_float3(7.2f, -2.0f, 22.3f), 1.44, 150, 0, 0.8f, 0, 4.8e6, -10.0},
    {"H2", make_float3(4.3f, -2.0f, 22.3f), 1.44, 250, 0, 1.2f, 0, 12.9e6, -10.0},
    {"G1", make_float3(8.7f, 112.0f, 5.0f), 0.25, 60, 1, 40.0f, 0, 6.9e6, 5.8},   // Z: G1+G2 combined
    {"G2", make_float3(9.7f, 117.0f, 0.5f), 0.25, 60, 1, 40.0f, 0, 6.0e6, 5.8},
    {"EM1", make_float3(kXP, 100.5f, 0), 0.225, 172, 0, 1.0f, 1, 9.48e7, -10.0},
    {"EM2", make_float3(kXP, 97.5f, 0), 0.225, 211, 0, 1.0f, 1, 9.39e7, -10.0},
    {"EM3", make_float3(kXP, 88.0f, 0), 0.225, 211, 0, 1.0f, 1, 2.90e7, -10.0},
    {"EM4", make_float3(kXP, 85.0f, 0), 0.225, 79, 0, 1.0f, 1, 9.87e6, -10.0},
    {"Charpak", make_float3(kXP, 93.0f, 0), 0.5, 158, 0, 1.0f, 1, 73.4e6, -10.0},
    {"Joliot", make_float3(kXP, 95.5f, 0), 0.25, 140, 0, 1.0f, 1, 13.0e6, -10.0},
    {"Degennes", make_float3(kXP, 77.5f, 0), 0.5, 140, 0, 1.0f, 1, 29.8e6, -10.0}};
static const double kRatePerM2Day = 1.44e7;   // 1 cm^-2 min^-1 (PDG, sea level)
static const int kNF = 5;
static const double kF[kNF] = {0.1, 0.25, 0.5, 1.0, 2.0};
static const int kK = 9;
static const float kS0 = 0.92f, kDS = 0.02f;

// ---------------------------------------------------------------- geometry
static float pyr_half(float z) { return kPyrHalfBase * (1.0f - z / kHorig); }
static bool in_pyramid(float x, float y, float z) {
    if (z < 0.0f) return true;   // bedrock plateau under the pyramid (never crossed by an upward ray)
    if (z > kHtop) return false;
    const float h = pyr_half(z);
    return fabsf(x) <= h && fabsf(y) <= h;
}
static bool in_granite(float x, float y, float z) {
    return x >= -4.3f && x <= 9.2f && y >= -9.6f && y <= -1.4f && z >= 41.5f && z <= 65.0f;
}
static bool in_box(float x, float y, float z, float3 c, float3 h) {
    return fabsf(x - c.x) <= h.x && fabsf(y - c.y) <= h.y && fabsf(z - c.z) <= h.z;
}
// Known structures (air), ADR-011 s2
static bool known_air(float x, float y, float z) {
    const float dxp = fabsf(x - kXP);
    // descending corridor, entrance y = 101.8 down to floor z = -2
    if (dxp <= 0.525f && y <= 101.8f) {
        const float zf = zDC(y);
        if (zf >= -2.0f && z >= zf && z <= zf + 1.34f) return true;
    }
    // ascending corridor
    if (dxp <= 0.525f && y >= 42.5f && y <= 76.7f) {
        const float z0 = zDC(76.7f);
        const float zf = z0 + (76.7f - y) / (76.7f - 42.5f) * (21.7f - z0);
        if (z >= zf && z <= zf + 1.34f) return true;
    }
    // horizontal passage
    if (dxp <= 0.525f && y >= 2.6f && y <= 42.5f && z >= 21.7f && z <= 22.9f) return true;
    // Queen's Chamber and Niche tunnel
    if (x >= 2.0f && x <= 7.75f && y >= -2.6f && y <= 2.6f && z >= 21.7f && z <= 27.2f) return true;
    if (x >= 7.75f && x <= 13.5f && y >= -0.2f && y <= 1.3f && z >= 21.7f && z <= 23.7f) return true;
    // Grand Gallery
    if (y >= 0.6f && y <= 42.5f) {
        const float zf = 21.7f + (42.5f - y) * kTanGG;
        if (z >= zf && z <= zf + 8.6f) {
            const float hw = 0.5f * (2.1f - 1.1f * (z - zf) / 8.6f);
            if (dxp <= hw) return true;
        }
    }
    // antechamber
    if (dxp <= 0.75f && y >= -2.4f && y <= 0.6f && z >= 43.0f && z <= 46.8f) return true;
    // King's Chamber + relieving chambers
    if (x >= -2.77f && x <= 7.7f && y >= -8.1f && y <= -2.9f) {
        if (z >= 43.0f && z <= 48.8f) return true;
        for (float zk : {50.0f, 53.0f, 56.0f, 59.0f, 62.0f})
            if (z >= zk && z <= zk + 1.0f) return true;
    }
    // al-Ma'mun tunnel 2 x 2 m from (0,109.6,7) to (x_p,76.7,5.5)
    {
        const float ax = 0.0f, ay = 109.6f, az = 7.0f, bx = kXP, by = 76.7f, bz = 5.5f;
        const float ux = bx - ax, uy = by - ay;
        const float L2 = ux * ux + uy * uy;
        float t = ((x - ax) * ux + (y - ay) * uy) / L2;
        if (t >= 0.0f && t <= 1.0f) {
            const float px = ax + t * ux, py = ay + t * uy, pz = az + t * (bz - az);
            const float lat = fabsf((x - px) * (-uy) + (y - py) * ux) / sqrtf(L2);
            if (lat <= 1.0f && fabsf(z - pz) <= 1.0f) return true;
        }
    }
    return false;
}
static bool in_bv_inclined(float x, float y, float z) {
    if (fabsf(x - kBVc.x) > kBVh.x) return false;
    const float c = cosf(26.3f * 0.0174532925f), s = sinf(26.3f * 0.0174532925f);
    const float vy = y - kBVc.y, vz = z - kBVc.z;
    const float u = -c * vy + s * vz;   // long axis: rises to the south
    const float w = s * vy + c * vz;
    return fabsf(u) <= 15.0f && fabsf(w) <= 4.3f;
}

// ---------------------------------------------------------------- heterogeneity (ADR-011 s3)
static float gauss_hash(unsigned key) {
    const float u1 = 1.0f - hash_unit(key);
    const float u2 = hash_unit(key ^ 0x5bd1e995u);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}
struct Het {
    int seed = -1;
    std::vector<float> ztop, dc, bx, by, ox, oy;
};
static Het make_het(int seed) {
    Het h; h.seed = seed;
    if (seed < 0) return h;
    std::mt19937_64 rng(0x4B48554Full ^ (unsigned long long)(seed) * 0x9E3779B97F4A7C15ull);
    std::uniform_real_distribution<float> th(0.6f, 1.5f), bs(1.0f, 2.0f), u01(0.0f, 1.0f);
    std::normal_distribution<float> nc(0.0f, 0.02f);
    float z = -2.0f;
    while (z < 141.0f) {
        z += th(rng);
        h.ztop.push_back(z); h.dc.push_back(nc(rng));
        const float bx = bs(rng), by = bs(rng);
        h.bx.push_back(bx); h.by.push_back(by);
        h.ox.push_back(u01(rng) * bx); h.oy.push_back(u01(rng) * by);
    }
    return h;
}
static float het_factor(const Het& h, float x, float y, float z, unsigned vkey) {
    if (h.seed < 0) return 1.0f;
    auto it = std::lower_bound(h.ztop.begin(), h.ztop.end(), z);
    const int c = std::min<int>(int(it - h.ztop.begin()), int(h.ztop.size()) - 1);
    const int ib = (int)floorf((x - h.ox[c]) / h.bx[c]), jb = (int)floorf((y - h.oy[c]) / h.by[c]);
    const unsigned bkey = hash_u32((unsigned)c * 73856093u ^ (unsigned)(ib + 4096) * 19349663u ^
                                   (unsigned)(jb + 4096) * 83492791u ^ (unsigned)h.seed * 0x9E3779B1u);
    const float fb = 1.0f + 0.05f * gauss_hash(bkey);
    const float fv = 1.0f + 0.03f * gauss_hash(vkey ^ (unsigned)(h.seed) * 0x27d4eb2du);
    return (1.0f + h.dc[c]) * fb * fv;
}

static void parallel_for(int n, const std::function<void(int)>& f) {
    const int nt = std::max(1, std::min(16, (int)std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] { for (int i = t; i < n; i += nt) f(i); });
    for (auto& x : th) x.join();
}
static inline float vx(int i) { return kGridLo.x + (i + 0.5f) * kVox; }
static inline float vy(int j) { return kGridLo.y + (j + 0.5f) * kVox; }
static inline float vz(int k) { return kGridLo.z + (k + 0.5f) * kVox; }
static inline size_t vidx(int i, int j, int k) { return (size_t(k) * kNY + j) * kNX + i; }

// masonry (no air structures) for one heterogeneity realisation
static void build_base(const Het& h, std::vector<float>& rho) {
    rho.assign(size_t(kNX) * kNY * kNZ, kRhoAir);
    parallel_for(kNZ, [&](int k) {
        const float z = vz(k);
        for (int j = 0; j < kNY; ++j)
            for (int i = 0; i < kNX; ++i) {
                const float x = vx(i), y = vy(j);
                if (!in_pyramid(x, y, z)) continue;
                const size_t idx = vidx(i, j, k);
                const float r0 = in_granite(x, y, z) ? kRhoGranite : kRhoStone;
                rho[idx] = r0 * het_factor(h, x, y, z, (unsigned)idx * 0x9E3779B1u);
            }
    });
}
static std::vector<unsigned char> g_known;   // K0 air mask
static void build_known() {
    g_known.assign(size_t(kNX) * kNY * kNZ, 0);
    parallel_for(kNZ, [&](int k) {
        for (int j = 0; j < kNY; ++j)
            for (int i = 0; i < kNX; ++i)
                g_known[vidx(i, j, k)] = known_air(vx(i), vy(j), vz(k)) ? 1 : 0;
    });
}
static void carve_pred(std::vector<float>& rho, float3 lo, float3 hi, const std::function<bool(float, float, float)>& p) {
    const int i0 = std::max(0, int((lo.x - kGridLo.x) / kVox) - 1), i1 = std::min(kNX - 1, int((hi.x - kGridLo.x) / kVox) + 1);
    const int j0 = std::max(0, int((lo.y - kGridLo.y) / kVox) - 1), j1 = std::min(kNY - 1, int((hi.y - kGridLo.y) / kVox) + 1);
    const int k0 = std::max(0, int((lo.z - kGridLo.z) / kVox) - 1), k1 = std::min(kNZ - 1, int((hi.z - kGridLo.z) / kVox) + 1);
    for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) for (int i = i0; i <= i1; ++i)
        if (p(vx(i), vy(j), vz(k))) rho[vidx(i, j, k)] = kRhoAir;
}
struct KScene { int id; int seed; float3 blindc; };
static void build_scene(const std::vector<float>& base, const KScene& s, std::vector<float>& rho) {
    rho = base;
    for (size_t v = 0; v < rho.size(); ++v) if (g_known[v]) rho[v] = kRhoAir;
    auto boxp = [](float3 c, float3 h) { return [c, h](float x, float y, float z) { return in_box(x, y, z, c, h); }; };
    if (s.id == S_K1 || s.id == S_K12) carve_pred(rho, kBVc - kBVh, kBVc + kBVh, boxp(kBVc, kBVh));
    if (s.id == S_K1I) carve_pred(rho, kBVc - make_float3(1, 15, 15), kBVc + make_float3(1, 15, 15), in_bv_inclined);
    if (s.id == S_K2 || s.id == S_K12) carve_pred(rho, kNFCc - kNFCh, kNFCc + kNFCh, boxp(kNFCc, kNFCh));
    if (s.id == S_KB) carve_pred(rho, s.blindc - kBVh, s.blindc + kBVh, boxp(s.blindc, kBVh));
}

// ---------------------------------------------------------------- muon physics (copied from check_ark)
static double reyna_IV(double z) {
    const double l = std::log10(z);
    return 0.00253 * std::pow(z, -(0.2455 + 1.288 * l - 0.2555 * l * l + 0.0209 * l * l * l));
}
static const double kMmu = 0.10566, kA = 0.217, kB = 4.0e-4, kP0 = 0.2;
struct TTable { const float* t; int nc, nx; float c0, c1, xmax; };
__host__ __device__ inline float ttab(const TTable& T, float c, float X) {
    float fc = (c - T.c0) / (T.c1 - T.c0) * (T.nc - 1);
    float fx = X / T.xmax * (T.nx - 1);
    fc = fminf(fmaxf(fc, 0.0f), T.nc - 1.001f);
    fx = fminf(fmaxf(fx, 0.0f), T.nx - 1.001f);
    const int ic = (int)fc, ix = (int)fx;
    const float ac = fc - ic, ax = fx - ix;
    const float* r0 = T.t + size_t(ic) * T.nx;
    const float* r1 = r0 + T.nx;
    return (1 - ac) * ((1 - ax) * r0[ix] + ax * r0[ix + 1]) + ac * ((1 - ax) * r1[ix] + ax * r1[ix + 1]);
}
static std::vector<float> build_ttable(int nc, int nx, float c0, float c1, float xmax, double& jv_1gev, double& jv_p0) {
    std::vector<float> t(size_t(nc) * nx);
    const int np = 6000;
    const double lp0 = std::log(kP0), lp1 = std::log(1.0e5);
    std::vector<double> p(np), J(np);
    for (int ic = 0; ic < nc; ++ic) {
        const double c = c0 + (c1 - c0) * ic / (nc - 1);
        for (int i = 0; i < np; ++i) p[i] = std::exp(lp0 + (lp1 - lp0) * i / (np - 1));
        J[np - 1] = 0.0;
        for (int i = np - 2; i >= 0; --i) {
            const double f0 = c * c * c * reyna_IV(p[i] * c) * p[i];
            const double f1 = c * c * c * reyna_IV(p[i + 1] * c) * p[i + 1];
            J[i] = J[i + 1] + 0.5 * (f0 + f1) * (lp1 - lp0) / (np - 1);
        }
        auto Jat = [&](double pm) {
            if (pm <= p[0]) return J[0];
            if (pm >= p[np - 1]) return 0.0;
            const double f = (std::log(pm) - lp0) / (lp1 - lp0) * (np - 1);
            const int i = (int)f;
            const double a = f - i;
            return (1 - a) * J[i] + a * J[i + 1];
        };
        if (ic == nc - 1) { jv_1gev = Jat(1.0) * 1e4; jv_p0 = J[0] * 1e4; }
        const double E0 = std::sqrt(kP0 * kP0 + kMmu * kMmu) - kMmu;
        for (int ix = 0; ix < nx; ++ix) {
            const double X = xmax * ix / (nx - 1);
            const double Ein = (E0 + kA / kB) * std::exp(kB * X) - kA / kB;
            const double pin = std::sqrt((Ein + kMmu) * (Ein + kMmu) - kMmu * kMmu);
            t[size_t(ic) * nx + ix] = float(Jat(std::max(pin, kP0)) / J[0]);
        }
    }
    return t;
}
static float ttab_invert(const TTable& T, float c, float Tobs) {
    float lo = 0.0f, hi = T.xmax;
    if (Tobs >= ttab(T, c, 0.0f)) return 0.0f;
    if (Tobs <= ttab(T, c, hi)) return hi;
    for (int it = 0; it < 50; ++it) {
        const float mid = 0.5f * (lo + hi);
        if (ttab(T, c, mid) > Tobs) lo = mid; else hi = mid;
    }
    return 0.5f * (lo + hi);
}

// ---------------------------------------------------------------- kernels
__device__ __forceinline__ int sky_bin(const MuonSky& sky, float c, float az) {
    const float th = acosf(fminf(1.0f, fmaxf(-1.0f, c)));
    int ith = static_cast<int>(th / sky.th_max * sky.n_th);
    ith = ith < 0 ? 0 : (ith >= sky.n_th ? sky.n_th - 1 : ith);
    int iaz = static_cast<int>(az * 0.15915494f * sky.n_az);
    iaz = iaz < 0 ? 0 : (iaz >= sky.n_az ? sky.n_az - 1 : iaz);
    return ith * sky.n_az + iaz;
}
// Expected map (khufu copy of ark_expected + acceptance mask: rays in bins outside the
// detector's window are not marched; their bins are never used).
__global__ void khufu_expected(const VoxelMedium m, MuonSky sky, int n, float3 chamber, TTable tt, int K,
                               float s0, float ds, const unsigned char* acc, double* tsum,
                               unsigned long long* open) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned key = hash_u32(static_cast<unsigned>(i) * 0x9E3779B9u);
    float3 d; float c, az;
    sample_sky(false, sky.c_min, key, d, c, az);
    const int bin = sky_bin(sky, c, az);
    if (open) atomicAdd(&open[bin], 1ULL);
    if (!acc[bin]) return;
    const float X = march_medium(m, chamber, d);
    for (int k = 0; k < K; ++k)
        atomicAdd(&tsum[size_t(k) * sky.size() + bin], (double)ttab(tt, c, (s0 + k * ds) * X));
}
__global__ void khufu_bernoulli(const VoxelMedium m, MuonSky sky, int n, float3 chamber, TTable tt,
                                const unsigned char* acc, unsigned seed, unsigned long long* det) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned key = hash_u32(static_cast<unsigned>(i) * 0x9E3779B9u);
    float3 d; float c, az;
    sample_sky(false, sky.c_min, key, d, c, az);
    const int bin = sky_bin(sky, c, az);
    if (!acc[bin]) return;
    const float X = march_medium(m, chamber, d);
    if (hash_unit(key ^ 0x2545F491u ^ seed) < ttab(tt, c, X)) atomicAdd(&det[bin], 1ULL);
}
// bin-centre opacity of the K0 model
__global__ void khufu_xb(const VoxelMedium m, int n, const float3* p, const float3* d, float* X) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    X[i] = march_medium(m, p[i], d[i]);
}
// template scan: dF(c) = sum_b N0_b (r - 1) - d_b ln r, r = T(X_b - 2.2 L_b(c)) / T(X_b)
struct LBin { float3 p, d; float c, X0, T0, N0, dc; };
__global__ void khufu_scan(const LBin* bins, int b0, int b1, TTable tt, float3 lo, float3 st, int nx, int ny, int nz,
                           float3 half, double* dF) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny * nz) return;
    const int ix = t % nx, iy = (t / nx) % ny, iz = t / (nx * ny);
    const float3 c = make_float3(lo.x + ix * st.x, lo.y + iy * st.y, lo.z + iz * st.z);
    const float bl[3] = {c.x - half.x, c.y - half.y, c.z - half.z}, bh[3] = {c.x + half.x, c.y + half.y, c.z + half.z};
    double acc = 0.0;
    for (int b = b0; b < b1; ++b) {
        const LBin B = bins[b];
        const float pc[3] = {B.p.x, B.p.y, B.p.z}, dc[3] = {B.d.x, B.d.y, B.d.z};
        float t0 = 0.0f, t1 = 1e30f;
        bool miss = false;
        for (int a = 0; a < 3 && !miss; ++a) {
            if (fabsf(dc[a]) < 1e-9f) { if (pc[a] < bl[a] || pc[a] > bh[a]) miss = true; continue; }
            const float inv = 1.0f / dc[a];
            float ta = (bl[a] - pc[a]) * inv, tb = (bh[a] - pc[a]) * inv;
            if (ta > tb) { const float q = ta; ta = tb; tb = q; }
            t0 = fmaxf(t0, ta); t1 = fminf(t1, tb);
            if (t1 <= t0) miss = true;
        }
        if (miss) continue;
        const float L = t1 - t0;
        const float T1 = ttab(tt, B.c, fmaxf(0.0f, B.X0 - 2.2f * L));
        const double r = double(T1) / double(B.T0);
        acc += double(B.N0) * (r - 1.0) - double(B.dc) * log(r);
    }
    dF[t] += acc;   // launched in bin chunks (keeps each launch far below the Windows TDR limit)
}

// ---------------------------------------------------------------- helpers
static double secs(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
template <typename T>
static void write_bin(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}
struct Gpu {
    float* d_rho = nullptr;
    double* d_tsum = nullptr;
    unsigned long long* d_ull = nullptr;
    unsigned char* d_acc = nullptr;   // [det][bin]
    float* d_tab = nullptr;
    TTable tt{};
    MuonSky sky;
    int M = 1 << 23;
};
static VoxelMedium sim_medium(const Gpu& g) {
    VoxelMedium m;
    m.lo = kGridLo; m.voxel = kVox; m.nx = kNX; m.ny = kNY; m.nz = kNZ;
    m.mu_rock = kRhoStone; m.mu_air = kRhoAir; m.mu = g.d_rho;
    return m;
}
// Upload a volume; run the listed detectors; return per-m^2-per-day expected rates [det][k][bin].
static std::vector<double> run_scene(Gpu& g, const std::vector<float>& rho, const std::vector<int>& dets, int K,
                                     std::vector<unsigned long long>* open_out) {
    CUDA_CHECK(cudaMemcpy(g.d_rho, rho.data(), rho.size() * sizeof(float), cudaMemcpyHostToDevice));
    const int nb = g.sky.size();
    std::vector<double> out(size_t(kNDet) * K * nb, 0.0);
    const VoxelMedium m = sim_medium(g);
    bool first = true;
    for (int dI : dets) {
        CUDA_CHECK(cudaMemset(g.d_tsum, 0, size_t(K) * nb * sizeof(double)));
        const bool want_open = open_out && first;
        if (want_open) CUDA_CHECK(cudaMemset(g.d_ull, 0, nb * sizeof(unsigned long long)));
        khufu_expected<<<muon_grid(g.M, 256), 256>>>(m, g.sky, g.M, kDet[dI].p, g.tt, K, K == 1 ? 1.0f : kS0, kDS,
                                                      g.d_acc + size_t(dI) * nb, g.d_tsum, want_open ? g.d_ull : nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data() + size_t(dI) * K * nb, g.d_tsum, size_t(K) * nb * sizeof(double),
                              cudaMemcpyDeviceToHost));
        if (want_open) {
            open_out->resize(nb);
            CUDA_CHECK(cudaMemcpy(open_out->data(), g.d_ull, nb * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        }
        first = false;
    }
    for (double& v : out) v *= kRatePerM2Day / g.M;
    return out;
}

int main(int argc, char** argv) {
    const std::string out = argc > 1 ? argv[1] : "experiments/khufu";
    const int log2M = argc > 2 ? std::atoi(argv[2]) : 23;
    const int nSeeds = argc > 3 ? std::atoi(argv[3]) : 8;
    const bool do_recon = argc > 4 ? std::atoi(argv[4]) != 0 : true;
    auto T0 = std::chrono::steady_clock::now();
    std::printf("\nQuBLAR -- check_khufu (ADR-011 pre-registered validation; SYNTHETIC)\n\n");
    std::printf("  grid %dx%dx%d @ %.1f m; %d detectors; 2^%d rays per detector per scene; seeds 1..%d\n",
                kNX, kNY, kNZ, kVox, kNDet, log2M, nSeeds);
    for (int dI = 6; dI < kNDet; ++dI) kDet[dI].p.z = zDC(kDet[dI].p.y) + 0.4f;

    Gpu g;
    g.M = 1 << log2M;
    g.sky = imaging_sky();
    const int nb = g.sky.size();
    MuonBinnedData geom; geom.sky = g.sky;

    double jv1 = 0, jvp0 = 0;
    const int NC = 128, NXT = 4096;
    const float C0 = 0.30f, C1 = 1.0f, XMAX = 800.0f;
    std::vector<float> tab = build_ttable(NC, NXT, C0, C1, XMAX, jv1, jvp0);
    CUDA_CHECK(cudaMalloc(&g.d_tab, tab.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g.d_tab, tab.data(), tab.size() * sizeof(float), cudaMemcpyHostToDevice));
    g.tt = TTable{g.d_tab, NC, NXT, C0, C1, XMAX};
    TTable htt{tab.data(), NC, NXT, C0, C1, XMAX};
    std::printf("  flux (Reyna 2006): vertical J(>1 GeV) = %.1f m^-2 s^-1 sr^-1; J(>%.1f GeV/c) = %.1f\n", jv1, kP0, jvp0);
    std::printf("  vertical survival T(X): 50 mwe %.4f | 100 %.4f | 200 %.5f | 300 %.6f\n", ttab(htt, 1, 50),
                ttab(htt, 1, 100), ttab(htt, 1, 200), ttab(htt, 1, 300));

    // acceptance masks (bin-centre direction)
    std::vector<unsigned char> acc(size_t(kNDet) * nb, 0);
    for (int dI = 0; dI < kNDet; ++dI) {
        float3 ax = kConeAim - kDet[dI].p;
        const float an = sqrtf(ax.x * ax.x + ax.y * ax.y + ax.z * ax.z);
        ax = make_float3(ax.x / an, ax.y / an, ax.z / an);
        int na = 0;
        for (int b = 0; b < nb; ++b) {
            const float3 d = geom.bin_direction(b);
            bool ok;
            if (kDet[dI].cone) ok = (d.x * ax.x + d.y * ax.y + d.z * ax.z) >= cosf(kDet[dI].a * 0.0174532925f);
            else ok = d.z > 0 && fabsf(d.x / d.z) <= kDet[dI].a && fabsf(d.y / d.z) <= kDet[dI].a;
            acc[size_t(dI) * nb + b] = ok ? 1 : 0;
            na += ok;
        }
        std::printf("  %-8s (%6.2f, %7.2f, %6.2f) %5.3f m2 %4.0f d, %s, %d bins in acceptance\n", kDet[dI].name,
                    kDet[dI].p.x, kDet[dI].p.y, kDet[dI].p.z, kDet[dI].area, kDet[dI].days,
                    kDet[dI].cone ? "cone 40 deg" : "box |tan|", na);
    }
    CUDA_CHECK(cudaMalloc(&g.d_acc, acc.size()));
    CUDA_CHECK(cudaMemcpy(g.d_acc, acc.data(), acc.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&g.d_rho, size_t(kNX) * kNY * kNZ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_tsum, size_t(kK) * nb * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.d_ull, nb * sizeof(unsigned long long)));

    auto t0 = std::chrono::steady_clock::now();
    build_known();
    int failures = 0;
    // detector-voxel check (declared, not a failure: a detector voxel of stone adds <= 0.25 m of stone uniformly)
    std::vector<float> meanbase;
    build_base(make_het(-1), meanbase);
    {
        std::vector<float> k0;
        build_scene(meanbase, KScene{S_K0, -1, {}}, k0);
        for (int dI = 0; dI < kNDet; ++dI) {
            const int i = int((kDet[dI].p.x - kGridLo.x) / kVox), j = int((kDet[dI].p.y - kGridLo.y) / kVox),
                      k = int((kDet[dI].p.z - kGridLo.z) / kVox);
            const float r = k0[vidx(i, j, k)];
            if (r > 0.01f) std::printf("  note: %s sits in a stone voxel (rho %.2f) at 0.5 m voxelisation\n", kDet[dI].name, r);
        }
        size_t nair = 0; for (auto v : g_known) nair += v;
        std::printf("  known-structure air voxels: %zu (%.0f m3); base + mask in %.1f s\n", nair, nair * 0.125, secs(t0));
    }

    // ---------------- models (ensemble mean, K scales), all detectors ----------------
    t0 = std::chrono::steady_clock::now();
    std::vector<int> allD(kNDet), bvD, nfcD;
    std::iota(allD.begin(), allD.end(), 0);
    for (int dI = 0; dI < kNDet; ++dI) (kDet[dI].set == 0 ? bvD : nfcD).push_back(dI);
    std::vector<unsigned long long> open;
    std::vector<std::vector<double>> model(kNModel);   // [det][k][bin]
    for (int s = 0; s < kNModel; ++s) {
        std::vector<float> v;
        build_scene(meanbase, KScene{s, -1, {}}, v);
        model[s] = run_scene(g, v, allD, kK, s == 0 ? &open : nullptr);
        for (double& x : model[s]) x = std::max(x, 1e-12);
    }
    std::printf("  models: %d scenes x %d detectors x %d scales in %.1f s\n", kNModel, kNDet, kK, secs(t0));
    auto MOD = [&](int s, int dI, int k, int b) { return model[s][(size_t(dI) * kK + k) * nb + b]; };

    // ---------------- data realisations ----------------
    t0 = std::chrono::steady_clock::now();
    // data[scene][seed] -> [det][bin] rates (per m^2 per day) of that realisation
    std::vector<std::vector<std::vector<double>>> data(4, std::vector<std::vector<double>>(nSeeds));
    std::vector<double> data_k12_s1;
    for (int k = 0; k < nSeeds; ++k) {
        std::vector<float> base, v;
        build_base(make_het(k + 1), base);
        for (int s : {S_K0, S_K1, S_K1I, S_K2}) {
            build_scene(base, KScene{s, k + 1, {}}, v);
            const std::vector<int>& dl = (s == S_K0) ? allD : (s == S_K2 ? nfcD : bvD);
            data[s][k] = run_scene(g, v, dl, 1, nullptr);
        }
        if (k == 0) {
            build_scene(base, KScene{S_K12, 1, {}}, v);
            data_k12_s1 = run_scene(g, v, allD, 1, nullptr);
        }
    }
    std::printf("  realisations: 4 scenes x %d seeds (+K12 seed 1) in %.1f s\n", nSeeds, secs(t0));

    // ---------------- validation: Bernoulli vs expected map (K0 seed 1, NE1) ----------------
    {
        const int Mb = 1 << 24;
        std::vector<float> base, v;
        build_base(make_het(1), base);
        build_scene(base, KScene{S_K0, 1, {}}, v);
        CUDA_CHECK(cudaMemcpy(g.d_rho, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
        const VoxelMedium m = sim_medium(g);
        CUDA_CHECK(cudaMemset(g.d_tsum, 0, nb * sizeof(double)));
        khufu_expected<<<muon_grid(Mb, 256), 256>>>(m, g.sky, Mb, kDet[0].p, g.tt, 1, 1.0f, 0.0f, g.d_acc, g.d_tsum, nullptr);
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> ex(nb);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(ex.data(), g.d_tsum, nb * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(g.d_ull, 0, nb * sizeof(unsigned long long)));
        khufu_bernoulli<<<muon_grid(Mb, 256), 256>>>(m, g.sky, Mb, kDet[0].p, g.tt, g.d_acc, 777u, g.d_ull);
        CUDA_CHECK(cudaGetLastError());
        std::vector<unsigned long long> det(nb);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(det.data(), g.d_ull, nb * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        double chi2 = 0, se = 0, sd = 0; int ndf = 0;
        for (int b = 0; b < nb; ++b) {
            if (!acc[b]) continue;
            se += ex[b]; sd += double(det[b]);
            if (ex[b] < 5.0) continue;
            chi2 += (det[b] - ex[b]) * (det[b] - ex[b]) / ex[b];
            ++ndf;
        }
        const bool ok = ndf > 0 && chi2 / ndf >= 0.8 && chi2 / ndf <= 1.2;
        std::printf("  validation (K0 seed 1, NE1, 2^24 Bernoulli, seed 777): expected %.0f, detected %.0f, "
                    "chi2/ndf = %.3f (ndf %d) %s\n", se, sd, chi2 / ndf, ndf, ok ? "PASS" : "FAIL");
        FILE* fv = std::fopen((out + "/validation.txt").c_str(), "w");
        std::fprintf(fv, "K0 seed1 NE1 2^24 Bernoulli seed 777: expected %.1f detected %.0f chi2 %.2f ndf %d chi2/ndf %.4f %s\n",
                     se, sd, chi2, ndf, chi2 / ndf, ok ? "PASS" : "FAIL");
        std::fclose(fv);
        if (!ok) ++failures;
    }

    // ---------------- compact bin sets ----------------
    struct BSet { std::vector<int> det, bin; std::vector<double> w; };
    BSet sets[2];
    for (int s = 0; s < 2; ++s)
        for (int dI = 0; dI < kNDet; ++dI) {
            if (kDet[dI].set != s) continue;
            for (int b = 0; b < nb; ++b)
                if (acc[size_t(dI) * nb + b]) {
                    sets[s].det.push_back(dI); sets[s].bin.push_back(b);
                    sets[s].w.push_back(kDet[dI].area * kDet[dI].days);
                }
        }
    std::printf("  bins in the statistic: Big-Void set %zu, NFC set %zu\n", sets[0].det.size(), sets[1].det.size());

    // ---------------- statistic (ADR-010 s5 with exposure weights) ----------------
    t0 = std::chrono::steady_clock::now();
    struct Pair { int X, set; const char* name; };
    const Pair pairs[3] = {{S_K1, 0, "K0_vs_K1"}, {S_K1I, 0, "K0_vs_K1i"}, {S_K2, 1, "K0_vs_K2"}};
    // per set, scene, k: ln mu and sum w mu
    auto key = [](int set, int s, int k) { return (set * kNModel + s) * kK + k; };
    std::vector<std::vector<double>> lnmu(2 * kNModel * kK);
    std::vector<double> swmu(2 * kNModel * kK, 0.0);
    for (int st = 0; st < 2; ++st)
        for (int s = 0; s < kNModel; ++s)
            for (int k = 0; k < kK; ++k) {
                auto& L = lnmu[key(st, s, k)];
                const size_t n = sets[st].det.size();
                L.resize(n);
                double sw = 0;
                for (size_t i = 0; i < n; ++i) {
                    const double v = MOD(s, sets[st].det[i], k, sets[st].bin[i]);
                    L[i] = std::log(v);
                    sw += sets[st].w[i] * v;
                }
                swmu[key(st, s, k)] = sw;
            }
    auto Fgrid = [&](const std::vector<double>& d, int st, int s, int k, double f) {
        const auto& L = lnmu[key(st, s, k)];
        double sd = 0;
        for (size_t b = 0; b < d.size(); ++b) sd += d[b] * L[b];
        return f * swmu[key(st, s, k)] - sd;
    };
    auto Fs = [&](const std::vector<double>& d, int st, int s, double f, double sv) {
        double fr = (sv - kS0) / kDS;
        int k = std::min(kK - 2, std::max(0, (int)std::floor(fr)));
        const double t = std::min(1.0, std::max(0.0, fr - k));
        const double h00 = 2*t*t*t - 3*t*t + 1, h10 = t*t*t - 2*t*t + t, h01 = -2*t*t*t + 3*t*t, h11 = t*t*t - t*t;
        const double* L0 = lnmu[key(st, s, k)].data();
        const double* L1 = lnmu[key(st, s, k + 1)].data();
        const double* Lm = lnmu[key(st, s, std::max(0, k - 1))].data();
        const double* Lp = lnmu[key(st, s, std::min(kK - 1, k + 2))].data();
        const double w0 = (k == 0) ? 1.0 : 0.5, w1 = (k + 1 == kK - 1) ? 1.0 : 0.5;
        const std::vector<double>& W = sets[st].w;
        double sm = 0, sd = 0;
        for (size_t b = 0; b < d.size(); ++b) {
            const double m0 = w0 * (L1[b] - Lm[b]), m1 = w1 * (Lp[b] - L0[b]);
            const double L = h00 * L0[b] + h10 * m0 + h01 * L1[b] + h11 * m1;
            sm += W[b] * std::exp(L);
            sd += d[b] * L;
        }
        return f * sm - sd;
    };
    auto Fmin = [&](const std::vector<double>& d, int st, int s, double f) {
        double F[kK];
        for (int k = 0; k < kK; ++k) F[k] = Fgrid(d, st, s, k, f);
        int km = 0;
        for (int k = 1; k < kK; ++k) if (F[k] < F[km]) km = k;
        double a = kS0 + kDS * std::max(0, km - 1), c = kS0 + kDS * std::min(kK - 1, km + 1);
        const double gr = 0.6180339887498949;
        double x1 = c - gr * (c - a), x2 = a + gr * (c - a);
        double f1 = Fs(d, st, s, f, x1), f2 = Fs(d, st, s, f, x2);
        for (int it = 0; it < 30; ++it) {
            if (f1 < f2) { c = x2; x2 = x1; f2 = f1; x1 = c - gr * (c - a); f1 = Fs(d, st, s, f, x1); }
            else { a = x1; x1 = x2; f1 = f2; x2 = a + gr * (c - a); f2 = Fs(d, st, s, f, x2); }
        }
        return std::min(F[km], std::min(f1, f2));
    };
    auto draw = [&](const std::vector<double>& rate, int st, double f, unsigned long long tag) {
        std::mt19937_64 rng(tag);
        const size_t n = sets[st].det.size();
        std::vector<double> d(n);
        for (size_t i = 0; i < n; ++i) {
            const double lam = rate[size_t(sets[st].det[i]) * nb + sets[st].bin[i]] * sets[st].w[i] * f;
            d[i] = lam > 0 ? double(std::poisson_distribution<long long>(lam)(rng)) : 0.0;
        }
        return d;
    };
    std::vector<double> qA(size_t(3) * kNF * nSeeds), qX(size_t(3) * kNF * nSeeds);
    parallel_for(nSeeds * kNF, [&](int job) {
        const int k = job / kNF, e = job % kNF;
        const double f = kF[e];
        const unsigned long long base = 0x4B4855ull * 1000003ull + (k + 1) * 1009ull + e * 17ull;
        for (int st = 0; st < 2; ++st) {
            const std::vector<double> dA = draw(data[S_K0][k], st, f, base * 31 + st);
            const double FA = Fmin(dA, st, S_K0, f);
            for (int p = 0; p < 3; ++p) {
                if (pairs[p].set != st) continue;
                const size_t o = (size_t(p) * kNF + e) * nSeeds + k;
                qA[o] = 2.0 * (FA - Fmin(dA, st, pairs[p].X, f));
                const std::vector<double> dX = draw(data[pairs[p].X][k], st, f, base * 31 + 2 + pairs[p].X);
                qX[o] = 2.0 * (Fmin(dX, st, S_K0, f) - Fmin(dX, st, pairs[p].X, f));
            }
        }
    });
    std::printf("  statistic: 3 pairs x %d exposures x %d seeds in %.1f s\n\n", kNF, nSeeds, secs(t0));

    FILE* fq = std::fopen((out + "/q_values.csv").c_str(), "w");
    FILE* fs = std::fopen((out + "/summary_detection.csv").c_str(), "w");
    std::fprintf(fq, "pair,f,seed,q_under_K0,q_under_X\n");
    std::fprintf(fs, "pair,f,Z_emp,AUC,median_qX,mean_qK0,sd_qK0,Z_asimov_ideal,Z_asimov_nuis\n");
    std::printf("  %-10s %5s %10s %6s %12s %12s %10s %9s %9s\n", "pair", "f", "Z_emp", "AUC", "med q_X", "mean q_K0",
                "sd q_K0", "Z_ideal", "Z_nuis");
    int verdict_a[3] = {0, 0, 0};
    double zbest[3] = {0, 0, 0}, z1[3] = {0, 0, 0}, auc1[3] = {0, 0, 0};
    for (int p = 0; p < 3; ++p) {
        const int st = pairs[p].set, X = pairs[p].X;
        // Asimov at f = 1 (data = expected counts of X at s = 1)
        std::vector<double> mX(sets[st].det.size());
        double D1 = 0, Fsat = 0;
        for (size_t i = 0; i < mX.size(); ++i) {
            const double w = sets[st].w[i];
            const double a = w * MOD(S_K0, sets[st].det[i], 4, sets[st].bin[i]);
            mX[i] = w * MOD(X, sets[st].det[i], 4, sets[st].bin[i]);
            D1 += 2 * (a - mX[i] + mX[i] * std::log(mX[i] / a));
            Fsat += mX[i] - mX[i] * std::log(mX[i] / w);
        }
        const double Dn1 = 2.0 * (Fmin(mX, st, S_K0, 1.0) - Fsat);
        for (int e = 0; e < kNF; ++e) {
            std::vector<double> a(nSeeds), xx(nSeeds);
            for (int k = 0; k < nSeeds; ++k) {
                const size_t o = (size_t(p) * kNF + e) * nSeeds + k;
                a[k] = qA[o]; xx[k] = qX[o];
                std::fprintf(fq, "%s,%.2f,%d,%.6g,%.6g\n", pairs[p].name, kF[e], k + 1, a[k], xx[k]);
            }
            const double mA = std::accumulate(a.begin(), a.end(), 0.0) / nSeeds;
            double vA = 0; for (double v : a) vA += (v - mA) * (v - mA);
            const double sA = std::sqrt(vA / std::max(1, nSeeds - 1));
            std::vector<double> sx = xx; std::sort(sx.begin(), sx.end());
            const double med = nSeeds % 2 ? sx[nSeeds / 2] : 0.5 * (sx[nSeeds / 2 - 1] + sx[nSeeds / 2]);
            double auc = 0; for (double u : xx) for (double v : a) auc += (u > v) ? 1.0 : (u == v ? 0.5 : 0.0);
            auc /= double(nSeeds) * nSeeds;
            const double Z = sA > 0 ? (med - mA) / sA : 0.0;
            const double Zi = std::sqrt(std::max(0.0, kF[e] * D1)), Zn = std::sqrt(std::max(0.0, kF[e] * Dn1));
            std::fprintf(fs, "%s,%.2f,%.5g,%.4g,%.6g,%.6g,%.6g,%.5g,%.5g\n", pairs[p].name, kF[e], Z, auc, med, mA, sA, Zi, Zn);
            std::printf("  %-10s %5.2f %10.2f %6.3f %12.1f %12.2f %10.2f %9.1f %9.1f\n", pairs[p].name, kF[e], Z, auc, med,
                        mA, sA, Zi, Zn);
            if (kF[e] <= 1.0 && Z >= 5.0 && auc >= 0.99) verdict_a[p] = 1;
            if (kF[e] <= 1.0) zbest[p] = std::max(zbest[p], Z);
            if (kF[e] == 1.0) { z1[p] = Z; auc1[p] = auc; }
        }
    }
    std::fclose(fq); std::fclose(fs);

    // ---------------- localisation: template scan (ADR-011 s5) ----------------
    t0 = std::chrono::steady_clock::now();
    // bin-centre K0 opacity X_b for every accepted (det, bin)
    std::vector<float> Xb(size_t(kNDet) * nb, 0.0f);
    {
        std::vector<float> k0;
        build_scene(meanbase, KScene{S_K0, -1, {}}, k0);
        CUDA_CHECK(cudaMemcpy(g.d_rho, k0.data(), k0.size() * sizeof(float), cudaMemcpyHostToDevice));
        std::vector<float3> P(size_t(kNDet) * nb), D(size_t(kNDet) * nb);
        for (int dI = 0; dI < kNDet; ++dI)
            for (int b = 0; b < nb; ++b) { P[size_t(dI) * nb + b] = kDet[dI].p; D[size_t(dI) * nb + b] = geom.bin_direction(b); }
        float3 *dP, *dD; float* dX;
        CUDA_CHECK(cudaMalloc(&dP, P.size() * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&dD, D.size() * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&dX, Xb.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dP, P.data(), P.size() * sizeof(float3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dD, D.data(), D.size() * sizeof(float3), cudaMemcpyHostToDevice));
        const int n = int(P.size());
        khufu_xb<<<muon_grid(n, 256), 256>>>(sim_medium(g), n, dP, dD, dX);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(Xb.data(), dX, Xb.size() * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(dP); cudaFree(dD); cudaFree(dX);
        write_bin(out + "/xb_K0_bincentre.f32", Xb);
    }
    struct Grid3 { float3 lo, st; int nx, ny, nz; float3 half; };
    const Grid3 gBV{make_float3(kXP - 20, -30, 40), make_float3(1, 1, 1), 41, 81, 41, kBVh};
    const Grid3 gNFC{make_float3(kXP - 5, 80, 16), make_float3(0.25f, 0.5f, 0.25f), 41, 49, 49, kNFCh};
    LBin* d_bins = nullptr; double* d_dF = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bins, std::max(sets[0].det.size(), sets[1].det.size()) * sizeof(LBin)));
    CUDA_CHECK(cudaMalloc(&d_dF, size_t(41) * 81 * 49 * sizeof(double)));
    // uses ONLY the data counts and the K0 model -- never the truth
    auto scan = [&](int st, const std::vector<double>& d, const Grid3& G, std::vector<float>* dFout) {
        std::vector<LBin> B;
        for (size_t i = 0; i < d.size(); ++i) {
            const int dI = sets[st].det[i], b = sets[st].bin[i];
            LBin L;
            L.p = kDet[dI].p; L.d = geom.bin_direction(b); L.c = L.d.z;
            L.X0 = Xb[size_t(dI) * nb + b]; L.T0 = ttab(htt, L.c, L.X0);
            L.N0 = float(sets[st].w[i] * MOD(S_K0, dI, 4, b)); L.dc = float(d[i]);
            if (L.T0 <= 0.0f || L.N0 <= 0.0f) continue;
            B.push_back(L);
        }
        CUDA_CHECK(cudaMemcpy(d_bins, B.data(), B.size() * sizeof(LBin), cudaMemcpyHostToDevice));
        const int nc = G.nx * G.ny * G.nz;
        CUDA_CHECK(cudaMemset(d_dF, 0, nc * sizeof(double)));
        for (int b0 = 0; b0 < int(B.size()); b0 += 4096) {
            khufu_scan<<<muon_grid(nc, 128), 128>>>(d_bins, b0, std::min(int(B.size()), b0 + 4096), g.tt, G.lo, G.st,
                                                     G.nx, G.ny, G.nz, G.half, d_dF);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        std::vector<double> dF(nc);
        CUDA_CHECK(cudaMemcpy(dF.data(), d_dF, nc * sizeof(double), cudaMemcpyDeviceToHost));
        int best = 0;
        for (int t = 1; t < nc; ++t) if (dF[t] < dF[best]) best = t;
        if (dFout) { dFout->resize(nc); for (int t = 0; t < nc; ++t) (*dFout)[t] = float(dF[t]); }
        const int ix = best % G.nx, iy = (best / G.nx) % G.ny, iz = best / (G.nx * G.ny);
        return make_float3(G.lo.x + ix * G.st.x, G.lo.y + iy * G.st.y, G.lo.z + iz * G.st.z);
    };
    auto dist = [](float3 a, float3 b) { return std::sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z)); };
    FILE* fl = std::fopen((out + "/localisation.csv").c_str(), "w");
    std::fprintf(fl, "scene,seed,est_x,est_y,est_z,true_x,true_y,true_z,dist_m,tol_m,within\n");
    int nok[2] = {0, 0};
    std::vector<std::vector<double>> dloc1(nSeeds), dloc2(nSeeds);
    for (int k = 0; k < nSeeds; ++k) {
        const unsigned long long tag = 0x10CA1ull * 7919ull + (k + 1);
        dloc1[k] = draw(data[S_K1][k], 0, 1.0, tag * 3 + 1);
        dloc2[k] = draw(data[S_K2][k], 1, 1.0, tag * 3 + 2);
        std::vector<float> v1, v2;
        const float3 e1 = scan(0, dloc1[k], gBV, k == 0 ? &v1 : nullptr);
        const float3 e2 = scan(1, dloc2[k], gNFC, k == 0 ? &v2 : nullptr);
        if (k == 0) { write_bin(out + "/scan_dF_K1_s1.f32", v1); write_bin(out + "/scan_dF_K2_s1.f32", v2); }
        const double d1 = dist(e1, kBVc), d2 = dist(e2, kNFCc);
        nok[0] += d1 <= 5.0; nok[1] += d2 <= 1.0;
        std::fprintf(fl, "K1,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,5.0,%d\n", k + 1, e1.x, e1.y, e1.z, kBVc.x, kBVc.y, kBVc.z, d1, d1 <= 5.0);
        std::fprintf(fl, "K2,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,1.0,%d\n", k + 1, e2.x, e2.y, e2.z, kNFCc.x, kNFCc.y, kNFCc.z, d2, d2 <= 1.0);
        std::printf("  localisation seed %d: K1 est (%.1f, %.1f, %.1f) err %.2f m | K2 est (%.2f, %.2f, %.2f) err %.2f m\n",
                    k + 1, e1.x, e1.y, e1.z, d1, e2.x, e2.y, e2.z, d2);
    }
    // K1i data scanned with the horizontal template (reported, not decisive)
    for (int k = 0; k < nSeeds; ++k) {
        const std::vector<double> d = draw(data[S_K1I][k], 0, 1.0, 0x1C11ull * 7919ull + k);
        const float3 e = scan(0, d, gBV, nullptr);
        std::fprintf(fl, "K1i_horizontal_template,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,5.0,%d\n", k + 1, e.x, e.y, e.z,
                     kBVc.x, kBVc.y, kBVc.z, dist(e, kBVc), dist(e, kBVc) <= 5.0);
    }
    std::fclose(fl);
    const bool verdict_b = nok[0] >= 7 && nok[1] >= 7;
    std::printf("  localisation: K1 within 5 m in %d/%d seeds, K2 within 1 m in %d/%d seeds (%.1f s)\n", nok[0], nSeeds,
                nok[1], nSeeds, secs(t0));

    // ---------------- direction check (reported) ----------------
    {
        FILE* fd = std::fopen((out + "/direction_check.csv").c_str(), "w");
        std::fprintf(fd, "scene,seed,detector,n_bins_pull_gt2,angle_to_true_centre_deg\n");
        for (int which = 0; which < 2; ++which)
            for (int k = 0; k < nSeeds; ++k) {
                const int st = which;
                const std::vector<double>& d = which == 0 ? dloc1[k] : dloc2[k];
                const float3 tc = which == 0 ? kBVc : kNFCc;
                for (int dI = 0; dI < kNDet; ++dI) {
                    if (kDet[dI].set != st) continue;
                    double sx = 0, sy = 0, sz = 0; int n = 0;
                    for (size_t i = 0; i < d.size(); ++i) {
                        if (sets[st].det[i] != dI) continue;
                        const double N0 = sets[st].w[i] * MOD(S_K0, dI, 4, sets[st].bin[i]);
                        const double pull = (d[i] - N0) / std::sqrt(std::max(N0, 1e-9));
                        if (pull <= 2.0) continue;
                        const float3 u = geom.bin_direction(sets[st].bin[i]);
                        const double w = d[i] - N0;
                        sx += w * u.x; sy += w * u.y; sz += w * u.z; ++n;
                    }
                    float3 t = tc - kDet[dI].p;
                    const double tn = std::sqrt(t.x * t.x + t.y * t.y + t.z * t.z), sn = std::sqrt(sx*sx + sy*sy + sz*sz);
                    const double ang = n ? std::acos(std::min(1.0, (sx * t.x + sy * t.y + sz * t.z) / (sn * tn))) * 57.29578 : -1;
                    std::fprintf(fd, "%s,%d,%s,%d,%.2f\n", which == 0 ? "K1" : "K2", k + 1, kDet[dI].name, n, ang);
                }
            }
        std::fclose(fd);
    }

    // ---------------- comparison with the papers (ADR-011 s5, rule d) ----------------
    int verdict_d_ok = 0, verdict_d_n = 0;
    {
        FILE* fc = std::fopen((out + "/paper_comparison.csv").c_str(), "w");
        std::fprintf(fc, "instrument,scene,S,B,Z_reg_sim,Z_paper,Z_rule,Z_consistent,tracks_sim,tracks_paper,tracks_ratio,tracks_consistent,Z_asimov_ideal_det\n");
        std::printf("\n  paper comparison (f = 1, published area x days):\n");
        std::printf("  %-12s %6s %9s %10s %10s %12s %12s %7s\n", "instrument", "scene", "Z_reg", "paper", "Zok", "tracks_sim",
                    "tracks_pap", "ratio");
        auto zreg = [&](int X, const std::vector<int>& dl, double& S, double& B, double& Za) {
            S = 0; B = 0; Za = 0;
            for (int dI : dl) {
                const double w = kDet[dI].area * kDet[dI].days;
                double mx = 0;
                for (int b = 0; b < nb; ++b)
                    if (acc[size_t(dI) * nb + b]) mx = std::max(mx, w * (MOD(X, dI, 4, b) - MOD(S_K0, dI, 4, b)));
                for (int b = 0; b < nb; ++b) {
                    if (!acc[size_t(dI) * nb + b]) continue;
                    const double n0 = w * MOD(S_K0, dI, 4, b), nx = w * MOD(X, dI, 4, b);
                    Za += 2 * (n0 - nx + nx * std::log(nx / n0));
                    if (mx > 0 && nx - n0 >= 0.2 * mx) { S += nx - n0; B += n0; }
                }
            }
            Za = std::sqrt(std::max(0.0, Za));
            return B > 0 ? S / std::sqrt(B) : 0.0;
        };
        for (int X : {S_K1, S_K1I, S_K2}) {
            std::vector<std::vector<int>> groups;
            for (int dI = 0; dI < kNDet; ++dI) {
                if (kDet[dI].set != (X == S_K2 ? 1 : 0)) continue;
                if (dI == 5) continue;           // G2 is summed with G1
                if (dI == 4) groups.push_back({4, 5}); else groups.push_back({dI});
            }
            for (auto& gr : groups) {
                double S, B, Za;
                const double Z = zreg(X, gr, S, B, Za);
                const KDet& D = kDet[gr[0]];
                const double zp = D.paper_Z;
                const bool lower = zp < 0;
                const bool zok = lower ? (Z >= 5.0) : (Z / zp >= 0.5 && Z / zp <= 2.0);
                std::string nm = gr.size() == 2 ? "G1+G2" : D.name;
                for (int dI : gr) {
                    double tr = 0;
                    for (int b = 0; b < nb; ++b) if (acc[size_t(dI) * nb + b]) tr += kDet[dI].area * kDet[dI].days * MOD(S_K0, dI, 4, b);
                    const double ratio = tr / kDet[dI].paper_tracks;
                    const bool tok = ratio >= 0.5 && ratio <= 2.0;
                    std::fprintf(fc, "%s,%s,%.6g,%.6g,%.4g,%s%.1f,%s,%d,%.6g,%.6g,%.4g,%d,%.4g\n", kDet[dI].name, scene_name(X), S, B,
                                 Z, lower ? ">" : "", std::fabs(zp), lower ? "Z_sim>=5" : "0.5<=ratio<=2", zok, tr,
                                 kDet[dI].paper_tracks, ratio, tok, Za);
                    std::printf("  %-12s %6s %9.2f %9s%.1f %10s %12.3g %12.3g %7.2f\n",
                                (gr.size() == 2 ? (std::string(kDet[dI].name) + "(G1+G2)") : nm).c_str(), scene_name(X), Z,
                                lower ? ">" : " ", std::fabs(zp), zok ? "yes" : "NO", tr, kDet[dI].paper_tracks, ratio);
                    if (X != S_K1I) {
                        verdict_d_n += 1 + (dI == gr[0]);
                        verdict_d_ok += int(tok) + (dI == gr[0] ? int(zok) : 0);
                    }
                }
            }
        }
        std::fclose(fc);
    }

    // ---------------- blind check (ADR-011 s2 KB, rule c) ----------------
    int verdict_c = 0;
    double blind_err = -1;
    {
        t0 = std::chrono::steady_clock::now();
        std::random_device rd;
        const unsigned long long bseed = (static_cast<unsigned long long>(rd()) << 32) ^ rd();
        {
            FILE* f = std::fopen((out + "/blind_seed.txt").c_str(), "w");
            std::fprintf(f, "%llu\n", bseed);
            std::fclose(f);
        }
        std::mt19937_64 rng(bseed);
        std::uniform_real_distribution<float> ux(kXP - 12, kXP + 12), uy(-20, 40), uz(45, 80);
        const float sa = std::sin(std::atan(kHorig / kPyrHalfBase));
        float3 bc{};
        int tries = 0;
        for (;;) {
            ++tries;
            bc = make_float3(ux(rng), uy(rng), uz(rng));
            if (dist(bc, kBVc) < 10.0) continue;
            bool bad = false;
            for (int cx = -1; cx <= 1 && !bad; cx += 2) for (int cy = -1; cy <= 1 && !bad; cy += 2) for (int cz = -1; cz <= 1 && !bad; cz += 2) {
                const float x = bc.x + cx * kBVh.x, y = bc.y + cy * kBVh.y, z = bc.z + cz * kBVh.z;
                const float marg = (pyr_half(z) - std::max(fabsf(x), fabsf(y))) * sa;
                if (marg < 5.0f || kHtop - z < 5.0f) bad = true;
            }
            if (bad) continue;
            const float3 lo = bc - kBVh - make_float3(2, 2, 2), hi = bc + kBVh + make_float3(2, 2, 2);
            for (int k = std::max(0, int((lo.z - kGridLo.z) / kVox)); k <= std::min(kNZ - 1, int((hi.z - kGridLo.z) / kVox)) && !bad; ++k)
                for (int j = std::max(0, int((lo.y - kGridLo.y) / kVox)); j <= std::min(kNY - 1, int((hi.y - kGridLo.y) / kVox)) && !bad; ++j)
                    for (int i = std::max(0, int((lo.x - kGridLo.x) / kVox)); i <= std::min(kNX - 1, int((hi.x - kGridLo.x) / kVox)); ++i)
                        if (g_known[vidx(i, j, k)] && in_box(vx(i), vy(j), vz(k), bc, kBVh + make_float3(2, 2, 2))) { bad = true; break; }
            if (!bad) break;
        }
        std::vector<float> base, v;
        build_base(make_het(101), base);
        build_scene(base, KScene{S_KB, 101, bc}, v);
        const std::vector<double> rate = run_scene(g, v, bvD, 1, nullptr);
        const std::vector<double> d = draw(rate, 0, 1.0, 0xB11Dull);
        std::vector<float> vB;
        const float3 e = scan(0, d, gBV, &vB);   // data + K0 model only
        write_bin(out + "/scan_dF_KB.f32", vB);
        {
            FILE* f = std::fopen((out + "/blind_estimate.txt").c_str(), "w");
            std::fprintf(f, "estimate %.2f %.2f %.2f\nwritten_before_truth 1\nwall_time_s %.3f\n", e.x, e.y, e.z, secs(T0));
            std::fclose(f);
        }
        std::printf("\n  BLIND: estimate written to blind_estimate.txt: (%.1f, %.1f, %.1f)\n", e.x, e.y, e.z);
        // reveal
        blind_err = dist(e, bc);
        verdict_c = blind_err <= 5.0;
        {
            FILE* f = std::fopen((out + "/blind_truth.txt").c_str(), "w");
            std::fprintf(f, "truth %.3f %.3f %.3f\nrejection_tries %d\nerror_m %.3f\nwithin_5m %d\nwall_time_s %.3f\n", bc.x, bc.y, bc.z,
                         tries, blind_err, verdict_c, secs(T0));
            std::fclose(f);
        }
        std::printf("  BLIND: truth revealed after the estimate: (%.2f, %.2f, %.2f); error %.2f m -> %s (%.1f s)\n", bc.x, bc.y,
                    bc.z, blind_err, verdict_c ? "FOUND" : "NOT found", secs(t0));
        // blind maps for the figures
        std::vector<float> rb(size_t(kNDet) * nb, 0.0f);
        for (size_t i = 0; i < rate.size(); ++i) rb[i] = float(rate[i]);
        write_bin(out + "/rate_real_KB.f32", rb);
    }

    // ---------------- maps for the figures ----------------
    {
        for (int s = 0; s < kNModel; ++s) {
            std::vector<float> mm(size_t(kNDet) * nb);
            for (int dI = 0; dI < kNDet; ++dI) for (int b = 0; b < nb; ++b) mm[size_t(dI) * nb + b] = float(MOD(s, dI, 4, b));
            write_bin(out + "/rate_model_" + std::string(scene_name(s)) + ".f32", mm);
        }
        for (int s : {S_K0, S_K1, S_K2}) {
            std::vector<float> mm(size_t(kNDet) * nb);
            for (size_t i = 0; i < mm.size(); ++i) mm[i] = float(data[s][0][i]);
            write_bin(out + "/rate_real_s1_" + std::string(scene_name(s)) + ".f32", mm);
        }
        std::vector<float> of(nb);
        for (int b = 0; b < nb; ++b) of[b] = float(double(open[b]) / g.M);
        write_bin(out + "/open_fraction.f32", of);
        write_bin(out + "/acceptance.u8", acc);
        // structure point cloud (0.5 m voxel centres): 1 known, 2 BV horizontal, 3 BV inclined, 4 NFC
        std::vector<float> pc;
        for (int k = 0; k < kNZ; ++k) for (int j = 0; j < kNY; ++j) for (int i = 0; i < kNX; ++i) {
            const float x = vx(i), y = vy(j), z = vz(k);
            int lab = 0;
            if (g_known[vidx(i, j, k)] && in_pyramid(x, y, z)) lab = 1;
            else if (in_box(x, y, z, kBVc, kBVh)) lab = 2;
            else if (in_box(x, y, z, kNFCc, kNFCh)) lab = 4;
            if (lab) { pc.push_back(x); pc.push_back(y); pc.push_back(z); pc.push_back(float(lab)); }
            if (in_bv_inclined(x, y, z)) { pc.push_back(x); pc.push_back(y); pc.push_back(z); pc.push_back(3.0f); }
        }
        write_bin(out + "/structures_points.f32", pc);
        FILE* fd = std::fopen((out + "/detectors.csv").c_str(), "w");
        std::fprintf(fd, "name,x,y,z,area_m2,days,acceptance,a,set\n");
        for (int dI = 0; dI < kNDet; ++dI)
            std::fprintf(fd, "%s,%.3f,%.3f,%.3f,%.3f,%.0f,%s,%.2f,%s\n", kDet[dI].name, kDet[dI].p.x, kDet[dI].p.y, kDet[dI].p.z,
                         kDet[dI].area, kDet[dI].days, kDet[dI].cone ? "cone_deg" : "box_tan", kDet[dI].a, kDet[dI].set ? "NFC" : "BV");
        std::fclose(fd);
    }

    // ---------------- MLEM density recon (figure; not part of any rule) ----------------
    if (do_recon) {
        t0 = std::chrono::steady_clock::now();
        const float rv = 1.0f; const int rnx = 240, rny = 240, rnz = 142;
        const float kap = 0.01f;
        std::vector<float> prior(size_t(rnx) * rny * rnz), truth(prior.size());
        std::vector<char> domain(prior.size(), 0);
        parallel_for(rnz, [&](int kz) {
            for (int j = 0; j < rny; ++j) for (int i = 0; i < rnx; ++i) {
                const float x = kGridLo.x + (i + 0.5f) * rv, y = kGridLo.y + (j + 0.5f) * rv, z = kGridLo.z + (kz + 0.5f) * rv;
                const size_t v = (size_t(kz) * rny + j) * rnx + i;
                float r = kRhoAir;
                if (in_pyramid(x, y, z)) r = in_granite(x, y, z) ? kRhoGranite : kRhoStone;
                if (known_air(x, y, z)) r = kRhoAir;
                prior[v] = r;
                float t = r;
                if (in_box(x, y, z, kBVc, kBVh) || in_box(x, y, z, kNFCc, kNFCh)) t = kRhoAir;
                truth[v] = t;
                const bool inBV = x >= kXP - 20 && x <= kXP + 20 && y >= -30 && y <= 50 && z >= 40 && z <= 80;
                const bool inNF = x >= kXP - 5 && x <= kXP + 5 && y >= 80 && y <= 104 && z >= 16 && z <= 28;
                domain[v] = (in_pyramid(x, y, z) && (inBV || inNF)) ? 1 : 0;
            }
        });
        std::vector<float> mu0(prior.size());
        for (size_t v = 0; v < prior.size(); ++v) mu0[v] = kap * prior[v];
        VoxelMedium mg; mg.lo = kGridLo; mg.voxel = rv; mg.nx = rnx; mg.ny = rny; mg.nz = rnz;
        mg.mu_rock = kap * kRhoStone; mg.mu_air = kap * kRhoAir; mg.mu = mu0.data();
        const double W = 1.0e4;
        std::vector<MuonBinnedData> bd(kNDet);
        std::mt19937_64 rng(0xBEEF12ull);
        for (int dI = 0; dI < kNDet; ++dI) {
            const double w = kDet[dI].area * kDet[dI].days;
            bd[dI].sky = g.sky; bd[dI].open.assign(nb, 0); bd[dI].det.assign(nb, 0);
            for (int b = 0; b < nb; ++b) {
                if (!acc[size_t(dI) * nb + b]) continue;
                const double No = w * kRatePerM2Day * double(open[b]) / g.M;
                const double lam = data_k12_s1[size_t(dI) * nb + b] * w;
                const double dcount = lam > 0 ? double(std::poisson_distribution<long long>(lam)(rng)) : 0.0;
                if (No < 1.0) continue;
                const float c = bd[dI].bin_direction(b).z;
                const float Tobs = float(std::min(std::max(dcount, 0.5) / No, 1.0));
                const float X = ttab_invert(htt, c, Tobs);
                bd[dI].open[b] = (unsigned long long)std::llround(W * No);
                bd[dI].det[b] = (unsigned long long)std::max(1LL, std::llround(W * No * std::exp(-kap * X)));
            }
        }
        std::vector<MuonView> views;
        for (int dI = 0; dI < kNDet; ++dI) views.push_back({kDet[dI].p, &bd[dI]});
        std::vector<float> rec = mlem_transmission(mg, views, 20, mu0, domain);
        for (float& v : rec) v /= kap;
        write_bin(out + "/recon1m_K12_s1.f32", rec);
        write_bin(out + "/truth1m_K12.f32", truth);
        write_bin(out + "/domain1m.u8", domain);
        std::printf("\n  MLEM density recon (K12 seed 1, f = 1, 1 m grid, 20 iters) in %.1f s\n", secs(t0));
    }

    // ---------------- decisions ----------------
    std::printf("\n  DECISIONS (ADR-011 s6):\n");
    std::printf("    (a) K0 vs K1 (6 BV instruments): %s  (max Z_emp at f<=1 = %.2f; f=1: Z %.2f AUC %.3f)\n",
                verdict_a[0] ? "DETECTED" : "NOT detected", zbest[0], z1[0], auc1[0]);
    std::printf("        K0 vs K2 (7 NFC instruments): %s  (max Z_emp at f<=1 = %.2f; f=1: Z %.2f AUC %.3f)\n",
                verdict_a[2] ? "DETECTED" : "NOT detected", zbest[2], z1[2], auc1[2]);
    std::printf("        [K0 vs K1i reported: %s, max Z_emp at f<=1 = %.2f]\n", verdict_a[1] ? "detected" : "not detected", zbest[1]);
    std::printf("    (b) localisation: K1 %d/%d within 5 m, K2 %d/%d within 1 m -> %s\n", nok[0], nSeeds, nok[1], nSeeds,
                verdict_b ? "PASS" : "FAIL");
    std::printf("    (c) blind: error %.2f m -> %s\n", blind_err, verdict_c ? "PASS" : "FAIL");
    std::printf("    (d) comparison: %d/%d instrument checks consistent (see paper_comparison.csv)\n", verdict_d_ok, verdict_d_n);

    FILE* fm = std::fopen((out + "/meta.txt").c_str(), "w");
    std::fprintf(fm, "log2M %d\nseeds %d\nnb %d\nn_th %d\nn_az %d\nth_max %f\nndet %d\n", log2M, nSeeds, nb, g.sky.n_th, g.sky.n_az,
                 g.sky.th_max, kNDet);
    std::fprintf(fm, "scanBV lo %.2f %.2f %.2f step 1 1 1 n 41 81 41\nscanNFC lo %.2f %.2f %.2f step 0.25 0.5 0.25 n 41 49 49\n",
                 gBV.lo.x, gBV.lo.y, gBV.lo.z, gNFC.lo.x, gNFC.lo.y, gNFC.lo.z);
    std::fprintf(fm, "vol1m 240 240 142 lo -120 -120 -2 vox 1\njv_1gev %f\njv_p0 %f\n", jv1, jvp0);
    std::fprintf(fm, "verdict_a_K1 %d\nverdict_a_K2 %d\nverdict_a_K1i %d\nverdict_b %d\nnok_K1 %d\nnok_K2 %d\nverdict_c %d\nblind_err %.3f\nverdict_d %d/%d\n",
                 verdict_a[0], verdict_a[2], verdict_a[1], verdict_b, nok[0], nok[1], verdict_c, blind_err, verdict_d_ok, verdict_d_n);
    std::fclose(fm);

    cudaFree(d_bins); cudaFree(d_dF);
    cudaFree(g.d_rho); cudaFree(g.d_tsum); cudaFree(g.d_ull); cudaFree(g.d_tab); cudaFree(g.d_acc);
    std::printf("\n  total wall time %.1f s\n%s\n\n", secs(T0), failures ? "VALIDATION FAILURES PRESENT" : "validation checks passed");
    return failures ? 1 : 0;
}
