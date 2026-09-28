// =============================================================================
// QuBLAR -- check_ree_muo: exploration muography of a REE carbonatite from a drift
// =============================================================================
// ADR-014 (pre-registered; frozen hash in experiments/ree_muography/PREREG_SHA256.txt).
// SYNTHETIC. The muon machinery is COPIED from src/check_khufu.cu (ADR-011), which
// copied it from src/check_ark.cu (ADR-010); neither is modified: Reyna/CSDA survival
// table, hash-keyed expected maps with acceptance masks, Bernoulli cross-check,
// profile-likelihood statistic with the Hermite/golden density-scale profile,
// template-scan localisation (here with a positive density contrast), blind check,
// mlem_transmission for the density figure.
// New here: flat-surface host rock with a smooth heterogeneity field, a drift with
// four 1 m2 detectors, dense cube bodies with exact partial-volume weighting, the
// 48-cell sweep (depth x size x contrast), exposures in days, flux sanity numbers.
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

// ---------------------------------------------------------------- ADR-014 s2
static const double kRmPi = 3.14159265358979323846;
static const float kRmVox = 1.0f;
static const int kRmNX = 480, kRmNY = 420, kRmNZ = 204;
static const float3 kRmLo = make_float3(-240.0f, -210.0f, -204.0f);
static const float kRmAir = 0.0012f, kRmHost = 2.70f;
static const float kRmLat = 30.0f, kRmNodeSd = 0.015f, kRmIid = 0.03f;
static bool rm_in_drift(float x, float y, float z) { (void)x; return fabsf(y) <= 2.5f && z >= -201.0f && z <= -196.0f; }

static const int kRmND = 4;
static const float3 kRmDet[kRmND] = {make_float3(-30, 0, -200), make_float3(-10, 0, -200), make_float3(10, 0, -200),
                                     make_float3(30, 0, -200)};
static const double kRmArea = 1.0;
static const double kRmRatePerM2Day = 1.44e7;   // 1 cm^-2 min^-1 (open sky normalisation, as ADR-010/011)
static const int kRmNT = 5;
static const double kRmT[kRmNT] = {30, 45, 90, 180, 365};
static const int kRmIT180 = 3;
static const int kRmK = 9;
static const float kRmS0 = 0.92f, kRmDS = 0.02f;

static const float kRmDepth[4] = {40, 80, 120, 160};
static const float kRmL[4] = {5, 10, 20, 40};
static const float kRmDrho[3] = {0.05f, 0.25f, 0.60f};
static const int kRmNC = 48;
struct RmCell { float d, L, drho; int id, il, ic; };
static RmCell rm_cell(int c) {
    const int ic = c % 3, il = (c / 3) % 4, id = c / 12;
    return RmCell{kRmDepth[id], kRmL[il], kRmDrho[ic], id, il, ic};
}
static float3 rm_centre(float d) { return make_float3(0.0f, 20.0f, -d); }
static const int kRmExample = (2 * 4 + 2) * 3 + 2;   // d = 120, L = 20, +0.60

// ---------------------------------------------------------------- muon physics (copied verbatim from check_khufu / check_ark)
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

// ---------------------------------------------------------------- heterogeneity (ADR-014 s3)
static float rm_gauss_hash(unsigned key) {
    const float u1 = 1.0f - hash_unit(key);
    const float u2 = hash_unit(key ^ 0x5bd1e995u);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}
struct RmHet {
    int seed = -1;
    float ox = 0, oy = 0, oz = 0;
    int i0 = 0, j0 = 0, k0 = 0, ni = 0, nj = 0, nk = 0;
    std::vector<float> node;
};
static RmHet rm_make_het(int seed) {
    RmHet h; h.seed = seed;
    if (seed < 0) return h;
    std::mt19937_64 rng(0x52454D55ull ^ (unsigned long long)(seed) * 0x9E3779B97F4A7C15ull);
    std::uniform_real_distribution<float> u(0.0f, kRmLat);
    h.ox = u(rng); h.oy = u(rng); h.oz = u(rng);
    h.i0 = (int)std::floor((kRmLo.x + h.ox) / kRmLat) - 1;
    h.j0 = (int)std::floor((kRmLo.y + h.oy) / kRmLat) - 1;
    h.k0 = (int)std::floor((kRmLo.z + h.oz) / kRmLat) - 1;
    h.ni = (int)std::ceil((kRmLo.x + kRmNX * kRmVox + h.ox) / kRmLat) - h.i0 + 2;
    h.nj = (int)std::ceil((kRmLo.y + kRmNY * kRmVox + h.oy) / kRmLat) - h.j0 + 2;
    h.nk = (int)std::ceil((kRmLo.z + kRmNZ * kRmVox + h.oz) / kRmLat) - h.k0 + 2;
    std::normal_distribution<float> nd(0.0f, kRmNodeSd);
    h.node.resize(size_t(h.ni) * h.nj * h.nk);
    for (float& v : h.node) v = nd(rng);
    return h;
}
static float rm_smooth(const RmHet& h, float x, float y, float z) {
    if (h.seed < 0) return 0.0f;
    const float fx = (x + h.ox) / kRmLat - h.i0, fy = (y + h.oy) / kRmLat - h.j0, fz = (z + h.oz) / kRmLat - h.k0;
    const int i = (int)floorf(fx), j = (int)floorf(fy), k = (int)floorf(fz);
    const float tx = fx - i, ty = fy - j, tz = fz - k;
    auto N = [&](int a, int b, int c) { return h.node[(size_t(c) * h.nj + b) * h.ni + a]; };
    float v = 0.0f;
    for (int c = 0; c < 2; ++c) for (int b = 0; b < 2; ++b) for (int a = 0; a < 2; ++a)
        v += (a ? tx : 1 - tx) * (b ? ty : 1 - ty) * (c ? tz : 1 - tz) * N(i + a, j + b, k + c);
    return v;
}
static void rm_parallel_for(int n, const std::function<void(int)>& f) {
    const int nt = std::max(1, std::min(16, (int)std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] { for (int i = t; i < n; i += nt) f(i); });
    for (auto& x : th) x.join();
}
static inline float rvx(int i) { return kRmLo.x + (i + 0.5f) * kRmVox; }
static inline float rvy(int j) { return kRmLo.y + (j + 0.5f) * kRmVox; }
static inline float rvz(int k) { return kRmLo.z + (k + 0.5f) * kRmVox; }
static inline size_t rvidx(int i, int j, int k) { return (size_t(k) * kRmNY + j) * kRmNX + i; }

// host rock x heterogeneity + drift (air); seed < 0: the analyst's mean model
static void rm_build_base(const RmHet& h, std::vector<float>& rho, double* sd_smooth, double* sd_total) {
    rho.assign(size_t(kRmNX) * kRmNY * kRmNZ, kRmAir);
    std::vector<double> s1(kRmNZ, 0), s2(kRmNZ, 0), t1(kRmNZ, 0), t2(kRmNZ, 0), cnt(kRmNZ, 0);
    rm_parallel_for(kRmNZ, [&](int k) {
        const float z = rvz(k);
        for (int j = 0; j < kRmNY; ++j)
            for (int i = 0; i < kRmNX; ++i) {
                const float x = rvx(i), y = rvy(j);
                const size_t idx = rvidx(i, j, k);
                if (rm_in_drift(x, y, z)) continue;
                float f = 1.0f;
                if (h.seed >= 0) {
                    const float sm = rm_smooth(h, x, y, z);
                    const float iid = kRmIid * rm_gauss_hash((unsigned)idx * 0x9E3779B1u ^ (unsigned)(h.seed) * 0x27d4eb2du);
                    f = (1.0f + sm) * (1.0f + iid);
                    s1[k] += sm; s2[k] += double(sm) * sm; t1[k] += f - 1.0; t2[k] += double(f - 1.0) * (f - 1.0); cnt[k] += 1;
                }
                rho[idx] = kRmHost * f;
            }
    });
    if (sd_smooth && h.seed >= 0) {
        double S1 = 0, S2 = 0, T1 = 0, T2 = 0, C = 0;
        for (int k = 0; k < kRmNZ; ++k) { S1 += s1[k]; S2 += s2[k]; T1 += t1[k]; T2 += t2[k]; C += cnt[k]; }
        *sd_smooth = std::sqrt(std::max(0.0, S2 / C - (S1 / C) * (S1 / C)));
        *sd_total = std::sqrt(std::max(0.0, T2 / C - (T1 / C) * (T1 / C)));
    }
}
// cube body, exact partial volume: rho *= (1 + drho * frac / host)  (== (host + drho frac) x het)
static float rm_ov(float c, float h, float v) {
    const float lo = std::max(c - h, v - 0.5f * kRmVox), hi = std::min(c + h, v + 0.5f * kRmVox);
    return std::max(0.0f, hi - lo) / kRmVox;
}
static void rm_body_range(float3 c, float L, int& i0, int& i1, int& j0, int& j1, int& k0, int& k1) {
    const float h = 0.5f * L;
    i0 = std::max(0, int(std::floor((c.x - h - kRmLo.x) / kRmVox)) - 1); i1 = std::min(kRmNX - 1, int((c.x + h - kRmLo.x) / kRmVox) + 1);
    j0 = std::max(0, int(std::floor((c.y - h - kRmLo.y) / kRmVox)) - 1); j1 = std::min(kRmNY - 1, int((c.y + h - kRmLo.y) / kRmVox) + 1);
    k0 = std::max(0, int(std::floor((c.z - h - kRmLo.z) / kRmVox)) - 1); k1 = std::min(kRmNZ - 1, int((c.z + h - kRmLo.z) / kRmVox) + 1);
}
static void rm_apply_body(std::vector<float>& work, const std::vector<float>& base, float3 c, float L, float drho) {
    int i0, i1, j0, j1, k0, k1; rm_body_range(c, L, i0, i1, j0, j1, k0, k1);
    const float h = 0.5f * L;
    for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) for (int i = i0; i <= i1; ++i) {
        const float fr = rm_ov(c.x, h, rvx(i)) * rm_ov(c.y, h, rvy(j)) * rm_ov(c.z, h, rvz(k));
        if (fr <= 0.0f) continue;
        const size_t v = rvidx(i, j, k);
        work[v] = base[v] * (1.0f + drho * fr / kRmHost);
    }
}
static void rm_restore(std::vector<float>& work, const std::vector<float>& base, float3 c, float L) {
    int i0, i1, j0, j1, k0, k1; rm_body_range(c, L, i0, i1, j0, j1, k0, k1);
    for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) for (int i = i0; i <= i1; ++i) {
        const size_t v = rvidx(i, j, k); work[v] = base[v];
    }
}

// ---------------------------------------------------------------- kernels (khufu copies, renamed)
__device__ __forceinline__ int rm_sky_bin(const MuonSky& sky, float c, float az) {
    const float th = acosf(fminf(1.0f, fmaxf(-1.0f, c)));
    int ith = static_cast<int>(th / sky.th_max * sky.n_th);
    ith = ith < 0 ? 0 : (ith >= sky.n_th ? sky.n_th - 1 : ith);
    int iaz = static_cast<int>(az * 0.15915494f * sky.n_az);
    iaz = iaz < 0 ? 0 : (iaz >= sky.n_az ? sky.n_az - 1 : iaz);
    return ith * sky.n_az + iaz;
}
__global__ void rm_expected(const VoxelMedium m, MuonSky sky, int n, float3 chamber, TTable tt, int K,
                            float s0, float ds, const unsigned char* acc, double* tsum, unsigned long long* open,
                            unsigned i0 = 0u) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned key = hash_u32((static_cast<unsigned>(i) + i0) * 0x9E3779B9u);
    float3 d; float c, az;
    sample_sky(false, sky.c_min, key, d, c, az);
    const int bin = rm_sky_bin(sky, c, az);
    if (open) atomicAdd(&open[bin], 1ULL);
    if (!acc[bin]) return;
    const float X = march_medium(m, chamber, d);
    for (int k = 0; k < K; ++k)
        atomicAdd(&tsum[size_t(k) * sky.size() + bin], (double)ttab(tt, c, (s0 + k * ds) * X));
}
__global__ void rm_bernoulli(const VoxelMedium m, MuonSky sky, int n, float3 chamber, TTable tt,
                             const unsigned char* acc, unsigned seed, unsigned long long* det,
                             unsigned i0 = 0u) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned key = hash_u32((static_cast<unsigned>(i) + i0) * 0x9E3779B9u);
    float3 d; float c, az;
    sample_sky(false, sky.c_min, key, d, c, az);
    const int bin = rm_sky_bin(sky, c, az);
    if (!acc[bin]) return;
    const float X = march_medium(m, chamber, d);
    if (hash_unit(key ^ 0x2545F491u ^ seed) < ttab(tt, c, X)) atomicAdd(&det[bin], 1ULL);
}
__global__ void rm_xb(const VoxelMedium m, int n, const float3* p, const float3* d, float* X) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    X[i] = march_medium(m, p[i], d[i]);
}
// template scan: dF(c) = sum_b N0_b (r - 1) - d_b ln r, r = T(X_b + drho L_b(c)) / T(X_b)
struct RmBin { float3 p, d; float c, X0, T0, N0, dc; };
__global__ void rm_scan(const RmBin* bins, int b0, int b1, TTable tt, float3 lo, float3 st, int nx, int ny, int nz,
                        float3 half, float drho, double* dF) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny * nz) return;
    const int ix = t % nx, iy = (t / nx) % ny, iz = t / (nx * ny);
    const float3 c = make_float3(lo.x + ix * st.x, lo.y + iy * st.y, lo.z + iz * st.z);
    const float bl[3] = {c.x - half.x, c.y - half.y, c.z - half.z}, bh[3] = {c.x + half.x, c.y + half.y, c.z + half.z};
    double acc = 0.0;
    for (int b = b0; b < b1; ++b) {
        const RmBin B = bins[b];
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
        const float T1 = ttab(tt, B.c, B.X0 + drho * L);
        const double r = double(T1) / double(B.T0);
        acc += double(B.N0) * (r - 1.0) - double(B.dc) * log(r);
    }
    dF[t] += acc;
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
struct RmGpu {
    float* d_rho = nullptr;
    double* d_tsum = nullptr;
    unsigned long long* d_ull = nullptr;
    unsigned char* d_acc = nullptr;
    float* d_tab = nullptr;
    TTable tt{};
    MuonSky sky;
    int M = 1 << 23;
};
static VoxelMedium rm_medium(const RmGpu& g) {
    VoxelMedium m;
    m.lo = kRmLo; m.voxel = kRmVox; m.nx = kRmNX; m.ny = kRmNY; m.nz = kRmNZ;
    m.mu_rock = kRmHost; m.mu_air = kRmAir; m.mu = g.d_rho;
    return m;
}
// upload a volume; run all detectors; per-m^2-per-day expected rates [det][k][bin]
static std::vector<double> rm_run(RmGpu& g, const std::vector<float>& rho, int K, std::vector<unsigned long long>* open_out) {
    CUDA_CHECK(cudaMemcpy(g.d_rho, rho.data(), rho.size() * sizeof(float), cudaMemcpyHostToDevice));
    const int nb = g.sky.size();
    std::vector<double> out(size_t(kRmND) * K * nb, 0.0);
    const VoxelMedium m = rm_medium(g);
    for (int dI = 0; dI < kRmND; ++dI) {
        CUDA_CHECK(cudaMemset(g.d_tsum, 0, size_t(K) * nb * sizeof(double)));
        const bool want_open = open_out && dI == 0;
        if (want_open) CUDA_CHECK(cudaMemset(g.d_ull, 0, nb * sizeof(unsigned long long)));
        rm_expected<<<muon_grid(g.M, 256), 256>>>(m, g.sky, g.M, kRmDet[dI], g.tt, K, K == 1 ? 1.0f : kRmS0, kRmDS,
                                                   g.d_acc + size_t(dI) * nb, g.d_tsum, want_open ? g.d_ull : nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data() + size_t(dI) * K * nb, g.d_tsum, size_t(K) * nb * sizeof(double), cudaMemcpyDeviceToHost));
        if (want_open) {
            open_out->resize(nb);
            CUDA_CHECK(cudaMemcpy(open_out->data(), g.d_ull, nb * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        }
    }
    for (double& v : out) v *= kRmRatePerM2Day / g.M;
    return out;
}

int main(int argc, char** argv) {
    const std::string out = argc > 1 ? argv[1] : "experiments/ree_muography";
    const int log2M = argc > 2 ? std::atoi(argv[2]) : 23;
    const int nSeeds = argc > 3 ? std::atoi(argv[3]) : 8;
    const bool do_recon = argc > 4 ? std::atoi(argv[4]) != 0 : true;
    auto T0 = std::chrono::steady_clock::now();
    std::printf("\nQuBLAR -- check_ree_muo (ADR-014 pre-registered; SYNTHETIC)\n\n");
    std::printf("  grid %dx%dx%d @ %.1f m; %d detectors; 2^%d rays per detector per scene; seeds 1..%d; %d cells\n",
                kRmNX, kRmNY, kRmNZ, kRmVox, kRmND, log2M, nSeeds, kRmNC);

    RmGpu g;
    g.M = 1 << log2M;
    g.sky = imaging_sky();
    const int nb = g.sky.size();
    MuonBinnedData geom; geom.sky = g.sky;

    double jv1 = 0, jvp0 = 0;
    const int NC = 128, NXT = 8192;
    const float C0 = 0.30f, C1 = 1.0f, XMAX = 1500.0f;
    std::vector<float> tab = build_ttable(NC, NXT, C0, C1, XMAX, jv1, jvp0);
    CUDA_CHECK(cudaMalloc(&g.d_tab, tab.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g.d_tab, tab.data(), tab.size() * sizeof(float), cudaMemcpyHostToDevice));
    g.tt = TTable{g.d_tab, NC, NXT, C0, C1, XMAX};
    TTable htt{tab.data(), NC, NXT, C0, C1, XMAX};
    std::printf("  flux (Reyna 2006): vertical J(>1 GeV) = %.1f m^-2 s^-1 sr^-1; J(>%.1f GeV/c) = %.1f\n", jv1, kP0, jvp0);
    std::printf("  vertical survival T(X): 300 mwe %.3e | 529.2 %.3e | 540 %.3e | 800 %.3e | 1200 %.3e\n", ttab(htt, 1, 300),
                ttab(htt, 1, 529.2f), ttab(htt, 1, 540), ttab(htt, 1, 800), ttab(htt, 1, 1200));

    // ---------------- flux sanity (ADR-014 R1b) ----------------
    FILE* fx = std::fopen((out + "/flux_sanity.txt").c_str(), "w");
    const double cmin = g.sky.c_min;
    const double I0v = (1.0 / 60.0) / (2.0 * kRmPi * (1.0 - cmin * cmin * cmin) / 3.0);   // cm^-2 s^-1 sr^-1
    const double Isim540 = I0v * ttab(htt, 1.0f, 540.0f), Isim529 = I0v * ttab(htt, 1.0f, 529.2f);
    const double h = 0.54;
    const double MH1 = 8.60e-6 * std::exp(-h / 0.45) + 0.44e-6 * std::exp(-h / 0.87);
    const double MH4 = 67.97e-6 * std::exp(-h / 0.285) + 2.071e-6 * std::exp(-h / 0.698);
    double Itot = 0.0;
    {
        const int n = 20000;
        for (int i = 0; i < n; ++i) {
            const double c = cmin + (1.0 - cmin) * (i + 0.5) / n;
            Itot += I0v * c * c * ttab(htt, float(c), float(540.0 / c)) * 2.0 * kRmPi * (1.0 - cmin) / n;
        }
    }
    std::fprintf(fx, "open_vertical_intensity_cm2_s_sr %.6e\nT_vertical_540mwe %.6e\nsim_vertical_540_cm2_s_sr %.6e\n"
                     "sim_vertical_529p2_scene_cm2_s_sr %.6e\nMeiHime_eq1_vertical_0p54 %.6e\nratio_sim_over_MH1 %.4f\n"
                     "sim_total_flux_flat_540_70deg_cm2_s %.6e\nMeiHime_eq4_total_0p54 %.6e\nratio_total_sim_over_MH4 %.4f\n",
                 I0v, ttab(htt, 1.0f, 540.0f), Isim540, Isim529, MH1, Isim540 / MH1, Itot, MH4, Itot / MH4);
    std::printf("  flux sanity: sim vertical at 540 mwe %.3e vs Mei-Hime eq.(1) %.3e (ratio %.2f); total %.3e vs eq.(4) %.3e (ratio %.2f)\n",
                Isim540, MH1, Isim540 / MH1, Itot, MH4, Itot / MH4);

    // acceptance masks (bin-centre direction), box |tan| <= 1
    std::vector<unsigned char> acc(size_t(kRmND) * nb, 0);
    for (int dI = 0; dI < kRmND; ++dI) {
        int na = 0;
        for (int b = 0; b < nb; ++b) {
            const float3 d = geom.bin_direction(b);
            const bool ok = d.z > 0 && fabsf(d.x / d.z) <= 1.0f && fabsf(d.y / d.z) <= 1.0f;
            acc[size_t(dI) * nb + b] = ok ? 1 : 0;
            na += ok;
        }
        std::printf("  D%d (%6.1f, %5.1f, %7.1f) %.1f m2, box |tan| <= 1, %d bins in acceptance\n", dI + 1, kRmDet[dI].x,
                    kRmDet[dI].y, kRmDet[dI].z, kRmArea, na);
    }
    CUDA_CHECK(cudaMalloc(&g.d_acc, acc.size()));
    CUDA_CHECK(cudaMemcpy(g.d_acc, acc.data(), acc.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&g.d_rho, size_t(kRmNX) * kRmNY * kRmNZ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_tsum, size_t(kRmK) * nb * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.d_ull, nb * sizeof(unsigned long long)));

    // ---------------- models: ensemble mean, K scales ----------------
    auto t0 = std::chrono::steady_clock::now();
    int failures = 0;
    std::vector<float> meanbase, work;
    rm_build_base(rm_make_het(-1), meanbase, nullptr, nullptr);
    const int NS = 1 + kRmNC;   // scene 0 = K0, scene 1 + c = cell c
    std::vector<unsigned long long> open;
    std::vector<std::vector<double>> model(NS);
    work = meanbase;
    for (int s = 0; s < NS; ++s) {
        if (s > 0) { const RmCell C = rm_cell(s - 1); rm_apply_body(work, meanbase, rm_centre(C.d), C.L, C.drho); }
        model[s] = rm_run(g, work, kRmK, s == 0 ? &open : nullptr);
        if (s > 0) { const RmCell C = rm_cell(s - 1); rm_restore(work, meanbase, rm_centre(C.d), C.L); }
        for (double& x : model[s]) x = std::max(x, 1e-15);
    }
    std::printf("  models: %d scenes x %d detectors x %d scales in %.1f s\n", NS, kRmND, kRmK, secs(t0));
    auto MOD = [&](int s, int dI, int k, int b) { return model[s][(size_t(dI) * kRmK + k) * nb + b]; };
    {
        double cpd[kRmND] = {0, 0, 0, 0}, rv = 0, ov = 0;
        for (int dI = 0; dI < kRmND; ++dI)
            for (int b = 0; b < nb; ++b) if (acc[size_t(dI) * nb + b]) cpd[dI] += kRmArea * MOD(0, dI, 4, b);
        // empirical vertical intensity: bins with theta < 5 deg at D2, rate / solid angle
        for (int b = 0; b < nb; ++b) {
            const int ith = b / g.sky.n_az;
            const double th0 = double(ith) / g.sky.n_th * g.sky.th_max, th1 = double(ith + 1) / g.sky.n_th * g.sky.th_max;
            if (th1 > 5.0 * kRmPi / 180.0 + 1e-9) continue;
            rv += MOD(0, 1, 4, b);
            ov += (std::cos(th0) - std::cos(th1)) * 2.0 * kRmPi / g.sky.n_az;
        }
        const double Iemp = rv / ov / 1e4 / 86400.0;
        std::fprintf(fx, "K0_mean_counts_per_day_in_acceptance D1 %.2f D2 %.2f D3 %.2f D4 %.2f\n", cpd[0], cpd[1], cpd[2], cpd[3]);
        std::fprintf(fx, "K0_mean_empirical_vertical_intensity_theta_lt_5deg_D2_cm2_s_sr %.6e\n", Iemp);
        std::printf("  K0 counts/day in acceptance: D1 %.1f D2 %.1f D3 %.1f D4 %.1f; empirical vertical (theta<5 deg, D2) %.3e\n",
                    cpd[0], cpd[1], cpd[2], cpd[3], Iemp);
    }

    // ---------------- data realisations ----------------
    t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<std::vector<double>>> data(NS, std::vector<std::vector<double>>(nSeeds));
    std::vector<float> base;
    FILE* fh = std::fopen((out + "/heterogeneity.csv").c_str(), "w");
    std::fprintf(fh, "seed,sd_smooth,sd_total\n");
    for (int k = 0; k < nSeeds; ++k) {
        double sds = 0, sdt = 0;
        rm_build_base(rm_make_het(k + 1), base, &sds, &sdt);
        std::fprintf(fh, "%d,%.5f,%.5f\n", k + 1, sds, sdt);
        work = base;
        for (int s = 0; s < NS; ++s) {
            if (s > 0) { const RmCell C = rm_cell(s - 1); rm_apply_body(work, base, rm_centre(C.d), C.L, C.drho); }
            data[s][k] = rm_run(g, work, 1, nullptr);
            if (s > 0) { const RmCell C = rm_cell(s - 1); rm_restore(work, base, rm_centre(C.d), C.L); }
        }
        std::printf("  seed %d: sd(smooth) %.4f sd(total) %.4f; %d scenes done (%.1f s)\n", k + 1, sds, sdt, NS, secs(t0));
    }
    std::fclose(fh);
    std::printf("  realisations: %d scenes x %d seeds in %.1f s\n", NS, nSeeds, secs(t0));

    // ---------------- validation: Bernoulli vs expected map (K0 seed 1, D2) ----------------
    // Frozen: 2^24 candidate rays. Survival ~1e-4 leaves only a few hundred detections, so accepted bins are
    // pooled greedily in bin order (zenith-ring major) into cells of >= 5 expected (rule fixed before any run).
    // Exploratory (not a rule): the same check with 2^28 rays in 16 chunks of 2^24.
    auto bern = [&](int log2n, double& chi2, int& ndf, double& se, double& sd) {
        const int Mb = 1 << 24, nch = 1 << std::max(0, log2n - 24);
        rm_build_base(rm_make_het(1), base, nullptr, nullptr);
        CUDA_CHECK(cudaMemcpy(g.d_rho, base.data(), base.size() * sizeof(float), cudaMemcpyHostToDevice));
        const VoxelMedium m = rm_medium(g);
        const unsigned char* accD2 = g.d_acc + size_t(1) * nb;
        CUDA_CHECK(cudaMemset(g.d_tsum, 0, nb * sizeof(double)));
        CUDA_CHECK(cudaMemset(g.d_ull, 0, nb * sizeof(unsigned long long)));
        for (int ch = 0; ch < nch; ++ch) {
            rm_expected<<<muon_grid(Mb, 256), 256>>>(m, g.sky, Mb, kRmDet[1], g.tt, 1, 1.0f, 0.0f, accD2, g.d_tsum, nullptr,
                                                      unsigned(ch) * unsigned(Mb));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            rm_bernoulli<<<muon_grid(Mb, 256), 256>>>(m, g.sky, Mb, kRmDet[1], g.tt, accD2, 777u, g.d_ull, unsigned(ch) * unsigned(Mb));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        std::vector<double> ex(nb);
        std::vector<unsigned long long> det(nb);
        CUDA_CHECK(cudaMemcpy(ex.data(), g.d_tsum, nb * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(det.data(), g.d_ull, nb * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        chi2 = 0; se = 0; sd = 0; ndf = 0;
        double e = 0, o = 0;
        for (int b = 0; b < nb; ++b) {
            if (!acc[size_t(1) * nb + b]) continue;
            e += ex[b]; o += double(det[b]); se += ex[b]; sd += double(det[b]);
            if (e >= 5.0) { chi2 += (o - e) * (o - e) / e; ++ndf; e = 0; o = 0; }
        }
    };
    {
        double chi2, se, sd; int ndf;
        bern(24, chi2, ndf, se, sd);
        const bool ok = ndf > 0 && chi2 / ndf >= 0.8 && chi2 / ndf <= 1.2;
        std::printf("  validation (K0 seed 1, D2, 2^24 Bernoulli, seed 777; pooled cells e>=5): expected %.1f, detected %.0f, "
                    "chi2/ndf = %.3f (ndf %d) %s\n", se, sd, chi2 / std::max(1, ndf), ndf, ok ? "PASS" : "FAIL");
        FILE* fv = std::fopen((out + "/validation.txt").c_str(), "w");
        std::fprintf(fv, "FROZEN K0 seed1 D2 2^24 Bernoulli seed 777 (accepted bins pooled in bin order to e>=5): expected %.2f detected %.0f chi2 %.3f ndf %d chi2/ndf %.4f %s\n",
                     se, sd, chi2, ndf, chi2 / std::max(1, ndf), ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
        double c2b, seb, sdb; int nb2;
        bern(28, c2b, nb2, seb, sdb);
        std::printf("  exploratory validation, 2^28 rays: expected %.1f, detected %.0f, chi2/ndf = %.3f (ndf %d)\n", seb, sdb,
                    c2b / std::max(1, nb2), nb2);
        std::fprintf(fv, "EXPLORATORY same with 2^28 rays (16 chunks): expected %.2f detected %.0f chi2 %.3f ndf %d chi2/ndf %.4f\n",
                     seb, sdb, c2b, nb2, c2b / std::max(1, nb2));
        std::fclose(fv);
    }

    // ---------------- compact bin set: all accepted bins of D1-D4 ----------------
    std::vector<int> sdet, sbin; std::vector<double> sw;
    for (int dI = 0; dI < kRmND; ++dI)
        for (int b = 0; b < nb; ++b)
            if (acc[size_t(dI) * nb + b]) { sdet.push_back(dI); sbin.push_back(b); sw.push_back(kRmArea); }
    const size_t NB = sdet.size();
    std::printf("  bins in the statistic: %zu\n", NB);

    // ---------------- statistic (ADR-011 s5, weights area x days) ----------------
    t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<double>> lnmu(size_t(NS) * kRmK);
    std::vector<double> swmu(size_t(NS) * kRmK, 0.0);
    for (int s = 0; s < NS; ++s)
        for (int k = 0; k < kRmK; ++k) {
            auto& L = lnmu[size_t(s) * kRmK + k];
            L.resize(NB);
            double a = 0;
            for (size_t i = 0; i < NB; ++i) { const double v = MOD(s, sdet[i], k, sbin[i]); L[i] = std::log(v); a += sw[i] * v; }
            swmu[size_t(s) * kRmK + k] = a;
        }
    auto Fgrid = [&](const std::vector<double>& d, int s, int k, double f) {
        const auto& L = lnmu[size_t(s) * kRmK + k];
        double sd = 0;
        for (size_t b = 0; b < d.size(); ++b) sd += d[b] * L[b];
        return f * swmu[size_t(s) * kRmK + k] - sd;
    };
    auto Fs = [&](const std::vector<double>& d, int s, double f, double sv) {
        double fr = (sv - kRmS0) / kRmDS;
        int k = std::min(kRmK - 2, std::max(0, (int)std::floor(fr)));
        const double t = std::min(1.0, std::max(0.0, fr - k));
        const double h00 = 2*t*t*t - 3*t*t + 1, h10 = t*t*t - 2*t*t + t, h01 = -2*t*t*t + 3*t*t, h11 = t*t*t - t*t;
        const double* L0 = lnmu[size_t(s) * kRmK + k].data();
        const double* L1 = lnmu[size_t(s) * kRmK + k + 1].data();
        const double* Lm = lnmu[size_t(s) * kRmK + std::max(0, k - 1)].data();
        const double* Lp = lnmu[size_t(s) * kRmK + std::min(kRmK - 1, k + 2)].data();
        const double w0 = (k == 0) ? 1.0 : 0.5, w1 = (k + 1 == kRmK - 1) ? 1.0 : 0.5;
        double sm = 0, sd = 0;
        for (size_t b = 0; b < d.size(); ++b) {
            const double m0 = w0 * (L1[b] - Lm[b]), m1 = w1 * (Lp[b] - L0[b]);
            const double L = h00 * L0[b] + h10 * m0 + h01 * L1[b] + h11 * m1;
            sm += sw[b] * std::exp(L);
            sd += d[b] * L;
        }
        return f * sm - sd;
    };
    auto Fmin = [&](const std::vector<double>& d, int s, double f) {
        double F[kRmK];
        for (int k = 0; k < kRmK; ++k) F[k] = Fgrid(d, s, k, f);
        int km = 0;
        for (int k = 1; k < kRmK; ++k) if (F[k] < F[km]) km = k;
        double a = kRmS0 + kRmDS * std::max(0, km - 1), c = kRmS0 + kRmDS * std::min(kRmK - 1, km + 1);
        const double gr = 0.6180339887498949;
        double x1 = c - gr * (c - a), x2 = a + gr * (c - a);
        double f1 = Fs(d, s, f, x1), f2 = Fs(d, s, f, x2);
        for (int it = 0; it < 30; ++it) {
            if (f1 < f2) { c = x2; x2 = x1; f2 = f1; x1 = c - gr * (c - a); f1 = Fs(d, s, f, x1); }
            else { a = x1; x1 = x2; f1 = f2; x2 = a + gr * (c - a); f2 = Fs(d, s, f, x2); }
        }
        return std::min(F[km], std::min(f1, f2));
    };
    auto draw = [&](const std::vector<double>& rate, double f, unsigned long long tag) {
        std::mt19937_64 rng(tag);
        std::vector<double> d(NB);
        for (size_t i = 0; i < NB; ++i) {
            const double lam = rate[size_t(sdet[i]) * nb + sbin[i]] * sw[i] * f;
            d[i] = lam > 0 ? double(std::poisson_distribution<long long>(lam)(rng)) : 0.0;
        }
        return d;
    };
    std::vector<double> qA(size_t(kRmNC) * kRmNT * nSeeds), qX(qA.size());
    rm_parallel_for(nSeeds * kRmNT, [&](int job) {
        const int k = job / kRmNT, e = job % kRmNT;
        const double f = kRmT[e];
        const unsigned long long bs = 0x524545ull * 1000003ull + (k + 1) * 1009ull + e * 17ull;
        const std::vector<double> dA = draw(data[0][k], f, bs * 31);
        const double FA = Fmin(dA, 0, f);
        for (int c = 0; c < kRmNC; ++c) {
            const size_t o = (size_t(c) * kRmNT + e) * nSeeds + k;
            qA[o] = 2.0 * (FA - Fmin(dA, 1 + c, f));
            const std::vector<double> dX = draw(data[1 + c][k], f, bs * 131 + 7 + c);
            qX[o] = 2.0 * (Fmin(dX, 0, f) - Fmin(dX, 1 + c, f));
        }
    });
    std::printf("  statistic: %d cells x %d exposures x %d seeds in %.1f s\n\n", kRmNC, kRmNT, nSeeds, secs(t0));

    FILE* fq = std::fopen((out + "/q_values.csv").c_str(), "w");
    FILE* fs = std::fopen((out + "/summary_detection.csv").c_str(), "w");
    std::fprintf(fq, "cell,d,L,drho,T_days,seed,q_under_K0,q_under_X\n");
    std::fprintf(fs, "cell,d,L,drho,T_days,Z_emp,AUC,median_qX,mean_qK0,sd_qK0,Z_asimov_ideal,Z_asimov_nuis,detected\n");
    std::vector<int> det180(kRmNC, 0), detAny(kRmNC, 0);
    std::printf("  %4s %5s %4s %5s | Z_emp (AUC) at T = 30 / 45 / 90 / 180 / 365 d | Z_ideal(180)\n", "cell", "d", "L", "drho");
    for (int c = 0; c < kRmNC; ++c) {
        const RmCell C = rm_cell(c);
        double D1 = 0, Fsat = 0;
        std::vector<double> mX(NB);
        for (size_t i = 0; i < NB; ++i) {
            const double a = sw[i] * MOD(0, sdet[i], 4, sbin[i]);
            mX[i] = sw[i] * MOD(1 + c, sdet[i], 4, sbin[i]);
            D1 += 2 * (a - mX[i] + mX[i] * std::log(mX[i] / a));
            Fsat += mX[i] - mX[i] * std::log(mX[i] / sw[i]);
        }
        const double Dn1 = 2.0 * (Fmin(mX, 0, 1.0) - Fsat);
        std::printf("  %4d %5.0f %4.0f %5.2f |", c, C.d, C.L, C.drho);
        double zi180 = 0;
        for (int e = 0; e < kRmNT; ++e) {
            std::vector<double> a(nSeeds), xx(nSeeds);
            for (int k = 0; k < nSeeds; ++k) {
                const size_t o = (size_t(c) * kRmNT + e) * nSeeds + k;
                a[k] = qA[o]; xx[k] = qX[o];
                std::fprintf(fq, "%d,%.0f,%.0f,%.2f,%.0f,%d,%.6g,%.6g\n", c, C.d, C.L, C.drho, kRmT[e], k + 1, a[k], xx[k]);
            }
            const double mA = std::accumulate(a.begin(), a.end(), 0.0) / nSeeds;
            double vA = 0; for (double v : a) vA += (v - mA) * (v - mA);
            const double sA = std::sqrt(vA / std::max(1, nSeeds - 1));
            std::vector<double> sx = xx; std::sort(sx.begin(), sx.end());
            const double med = nSeeds % 2 ? sx[nSeeds / 2] : 0.5 * (sx[nSeeds / 2 - 1] + sx[nSeeds / 2]);
            double auc = 0; for (double u : xx) for (double v : a) auc += (u > v) ? 1.0 : (u == v ? 0.5 : 0.0);
            auc /= double(nSeeds) * nSeeds;
            const double Z = sA > 0 ? (med - mA) / sA : 0.0;
            const double Zi = std::sqrt(std::max(0.0, kRmT[e] * D1)), Zn = std::sqrt(std::max(0.0, kRmT[e] * Dn1));
            const int detd = (Z >= 5.0 && auc >= 0.99) ? 1 : 0;
            if (kRmT[e] <= 180.0 && detd) detAny[c] = 1;
            if (e == kRmIT180) { det180[c] = detd; zi180 = Zi; }
            std::fprintf(fs, "%d,%.0f,%.0f,%.2f,%.0f,%.5g,%.4g,%.6g,%.6g,%.6g,%.5g,%.5g,%d\n", c, C.d, C.L, C.drho, kRmT[e], Z, auc, med,
                         mA, sA, Zi, Zn, detd);
            std::printf(" %7.2f (%.2f)", Z, auc);
        }
        std::printf(" | %.1f%s\n", zi180, detAny[c] ? "  DETECTED" : "");
    }
    std::fclose(fq); std::fclose(fs);

    // ---------------- localisation: template scan (ADR-014 s5) ----------------
    t0 = std::chrono::steady_clock::now();
    std::vector<float> Xb(size_t(kRmND) * nb, 0.0f);
    {
        CUDA_CHECK(cudaMemcpy(g.d_rho, meanbase.data(), meanbase.size() * sizeof(float), cudaMemcpyHostToDevice));
        std::vector<float3> P(size_t(kRmND) * nb), D(size_t(kRmND) * nb);
        for (int dI = 0; dI < kRmND; ++dI)
            for (int b = 0; b < nb; ++b) { P[size_t(dI) * nb + b] = kRmDet[dI]; D[size_t(dI) * nb + b] = geom.bin_direction(b); }
        float3 *dP, *dD; float* dX;
        CUDA_CHECK(cudaMalloc(&dP, P.size() * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&dD, D.size() * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&dX, Xb.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dP, P.data(), P.size() * sizeof(float3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dD, D.data(), D.size() * sizeof(float3), cudaMemcpyHostToDevice));
        const int n = int(P.size());
        rm_xb<<<muon_grid(n, 256), 256>>>(rm_medium(g), n, dP, dD, dX);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(Xb.data(), dX, Xb.size() * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(dP); cudaFree(dD); cudaFree(dX);
        write_bin(out + "/xb_K0_bincentre.f32", Xb);
    }
    struct Grid3 { float3 lo, st; int nx, ny, nz; };
    const Grid3 gCell{make_float3(-60, -40, -190), make_float3(2, 2, 2), 61, 61, 91};
    const Grid3 gBlind{make_float3(-70, -70, -190), make_float3(2, 2, 2), 71, 71, 91};
    RmBin* d_bins = nullptr; double* d_dF = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bins, NB * sizeof(RmBin)));
    CUDA_CHECK(cudaMalloc(&d_dF, size_t(71) * 71 * 91 * sizeof(double)));
    // uses ONLY the data counts, the K0 model and the hypothesis (L, drho) -- never the truth position
    auto scan = [&](const std::vector<double>& d, double Tdays, float L, float drho, const Grid3& G, std::vector<float>* dFout) {
        std::vector<RmBin> B;
        for (size_t i = 0; i < NB; ++i) {
            const int dI = sdet[i], b = sbin[i];
            RmBin R;
            R.p = kRmDet[dI]; R.d = geom.bin_direction(b); R.c = R.d.z;
            R.X0 = Xb[size_t(dI) * nb + b]; R.T0 = ttab(htt, R.c, R.X0);
            R.N0 = float(sw[i] * Tdays * MOD(0, dI, 4, b)); R.dc = float(d[i]);
            if (R.T0 <= 0.0f || R.N0 <= 0.0f) continue;
            B.push_back(R);
        }
        CUDA_CHECK(cudaMemcpy(d_bins, B.data(), B.size() * sizeof(RmBin), cudaMemcpyHostToDevice));
        const int nc = G.nx * G.ny * G.nz;
        CUDA_CHECK(cudaMemset(d_dF, 0, nc * sizeof(double)));
        const float3 half = make_float3(0.5f * L, 0.5f * L, 0.5f * L);
        for (int b0 = 0; b0 < int(B.size()); b0 += 4096) {
            rm_scan<<<muon_grid(nc, 128), 128>>>(d_bins, b0, std::min(int(B.size()), b0 + 4096), g.tt, G.lo, G.st, G.nx, G.ny, G.nz,
                                                  half, drho, d_dF);
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
    const double T180 = kRmT[kRmIT180];
    FILE* fl = std::fopen((out + "/localisation.csv").c_str(), "w");
    std::fprintf(fl, "cell,d,L,drho,seed,est_x,est_y,est_z,true_x,true_y,true_z,dist_m,tol_m,within,detected_T180\n");
    std::vector<int> nok(kRmNC, 0);
    std::vector<double> dEx;   // example cell seed 1 counts (for the excess maps / MLEM)
    for (int c = 0; c < kRmNC; ++c) {
        const RmCell C = rm_cell(c);
        const float3 tc = rm_centre(C.d);
        const double tol = std::max(5.0, 0.5 * C.L);
        for (int k = 0; k < nSeeds; ++k) {
            const unsigned long long tag = 0x10CA14ull * 7919ull + (unsigned long long)(c) * 101ull + (k + 1);
            const std::vector<double> d = draw(data[1 + c][k], T180, tag);
            std::vector<float> v;
            const float3 e = scan(d, T180, C.L, C.drho, gCell, (c == kRmExample && k == 0) ? &v : nullptr);
            if (c == kRmExample && k == 0) { write_bin(out + "/scan_dF_example_s1.f32", v); dEx = d; }
            const double dd = dist(e, tc);
            nok[c] += dd <= tol;
            std::fprintf(fl, "%d,%.0f,%.0f,%.2f,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.3f,%.1f,%d,%d\n", c, C.d, C.L, C.drho, k + 1, e.x, e.y,
                         e.z, tc.x, tc.y, tc.z, dd, tol, dd <= tol, det180[c]);
        }
        std::printf("  localisation cell %2d (d %3.0f, L %2.0f, %+.2f): %d/%d within %.1f m%s\n", c, C.d, C.L, C.drho, nok[c], nSeeds,
                    tol, det180[c] ? " [detected at 180 d]" : "");
    }
    std::fclose(fl);
    std::printf("  localisation scans done in %.1f s\n", secs(t0));
    // K0 seed 1 counts at 180 d (excess-map reference)
    const std::vector<double> dK0 = draw(data[0][0], T180, 0x10CA14ull * 7919ull + 999999ull);

    // ---------------- blind check (ADR-014 s5) ----------------
    int verdict_blind = 0;
    double blind_err = -1;
    std::vector<double> blind_rate;
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
        std::uniform_real_distribution<float> uxy(-50, 50), uz(-160, -60);
        float3 bc{};
        int tries = 0;
        for (;;) {
            ++tries;
            bc = make_float3(uxy(rng), uxy(rng), uz(rng));
            const float hh = bc.z - kRmDet[0].z;
            int nv = 0;
            for (int dI = 0; dI < kRmND; ++dI)
                if (fabsf(bc.x - kRmDet[dI].x) <= 0.8f * hh && fabsf(bc.y - kRmDet[dI].y) <= 0.8f * hh) ++nv;
            if (nv >= 2) break;
        }
        const float bL = 20.0f, bdr = 0.60f;
        rm_build_base(rm_make_het(101), base, nullptr, nullptr);
        work = base;
        rm_apply_body(work, base, bc, bL, bdr);
        blind_rate = rm_run(g, work, 1, nullptr);
        const std::vector<double> d = draw(blind_rate, T180, 0xB11D14ull);
        std::vector<float> vB;
        const float3 e = scan(d, T180, bL, bdr, gBlind, &vB);   // data + K0 model + (L, drho) only
        write_bin(out + "/scan_dF_blind.f32", vB);
        {
            FILE* f = std::fopen((out + "/blind_estimate.txt").c_str(), "w");
            std::fprintf(f, "estimate %.2f %.2f %.2f\nwritten_before_truth 1\nwall_time_s %.3f\n", e.x, e.y, e.z, secs(T0));
            std::fclose(f);
        }
        std::printf("\n  BLIND: estimate written to blind_estimate.txt: (%.1f, %.1f, %.1f)\n", e.x, e.y, e.z);
        blind_err = dist(e, bc);
        verdict_blind = blind_err <= 10.0;
        {
            FILE* f = std::fopen((out + "/blind_truth.txt").c_str(), "w");
            std::fprintf(f, "truth %.3f %.3f %.3f\nL %.1f\ndrho %.2f\nrejection_tries %d\nerror_m %.3f\nwithin_10m %d\nwall_time_s %.3f\n",
                         bc.x, bc.y, bc.z, bL, bdr, tries, blind_err, verdict_blind, secs(T0));
            std::fclose(f);
        }
        std::printf("  BLIND: truth revealed after the estimate: (%.2f, %.2f, %.2f); error %.2f m -> %s (%.1f s)\n", bc.x, bc.y, bc.z,
                    blind_err, verdict_blind ? "FOUND" : "NOT found", secs(t0));
        // blind counts for the excess map
        std::vector<float> cb(size_t(kRmND) * nb, -1.0f);
        for (size_t i = 0; i < NB; ++i) cb[size_t(sdet[i]) * nb + sbin[i]] = float(d[i]);
        write_bin(out + "/counts_blind_T180.f32", cb);
    }

    // ---------------- maps for the figures ----------------
    {
        auto rates = [&](int s) {
            std::vector<float> mm(size_t(kRmND) * nb);
            for (int dI = 0; dI < kRmND; ++dI) for (int b = 0; b < nb; ++b) mm[size_t(dI) * nb + b] = float(MOD(s, dI, 4, b));
            return mm;
        };
        write_bin(out + "/rate_model_K0.f32", rates(0));
        write_bin(out + "/rate_model_example.f32", rates(1 + kRmExample));
        std::vector<float> cE(size_t(kRmND) * nb, -1.0f), cK(size_t(kRmND) * nb, -1.0f);
        for (size_t i = 0; i < NB; ++i) {
            cE[size_t(sdet[i]) * nb + sbin[i]] = float(dEx[i]);
            cK[size_t(sdet[i]) * nb + sbin[i]] = float(dK0[i]);
        }
        write_bin(out + "/counts_example_s1_T180.f32", cE);
        write_bin(out + "/counts_K0_s1_T180.f32", cK);
        std::vector<float> rb(blind_rate.begin(), blind_rate.end());
        write_bin(out + "/rate_real_blind.f32", rb);
        std::vector<float> of(nb);
        for (int b = 0; b < nb; ++b) of[b] = float(double(open[b]) / g.M);
        write_bin(out + "/open_fraction.f32", of);
        write_bin(out + "/acceptance.u8", acc);
        FILE* fd = std::fopen((out + "/detectors.csv").c_str(), "w");
        std::fprintf(fd, "name,x,y,z,area_m2,acceptance\n");
        for (int dI = 0; dI < kRmND; ++dI)
            std::fprintf(fd, "D%d,%.2f,%.2f,%.2f,%.2f,box_tan_1\n", dI + 1, kRmDet[dI].x, kRmDet[dI].y, kRmDet[dI].z, kRmArea);
        std::fclose(fd);
        FILE* fc = std::fopen((out + "/cells.csv").c_str(), "w");
        std::fprintf(fc, "cell,d,L,drho,cx,cy,cz\n");
        for (int c = 0; c < kRmNC; ++c) {
            const RmCell C = rm_cell(c); const float3 tc = rm_centre(C.d);
            std::fprintf(fc, "%d,%.0f,%.0f,%.2f,%.1f,%.1f,%.1f\n", c, C.d, C.L, C.drho, tc.x, tc.y, tc.z);
        }
        std::fclose(fc);
    }

    // ---------------- MLEM density recon (figure; not part of any rule) ----------------
    if (do_recon) {
        t0 = std::chrono::steady_clock::now();
        const float rv = 2.0f; const int rnx = 240, rny = 210, rnz = 102;
        const float kap = 0.01f;
        const RmCell C = rm_cell(kRmExample);
        const float3 tc = rm_centre(C.d);
        std::vector<float> prior(size_t(rnx) * rny * rnz), truth(prior.size());
        std::vector<char> domain(prior.size(), 0);
        rm_parallel_for(rnz, [&](int kz) {
            for (int j = 0; j < rny; ++j) for (int i = 0; i < rnx; ++i) {
                const float x = kRmLo.x + (i + 0.5f) * rv, y = kRmLo.y + (j + 0.5f) * rv, z = kRmLo.z + (kz + 0.5f) * rv;
                const size_t v = (size_t(kz) * rny + j) * rnx + i;
                float r = rm_in_drift(x, y, z) ? kRmAir : kRmHost;
                prior[v] = r;
                const float h = 0.5f * C.L;
                auto ov2 = [&](float c0, float p) {
                    const float lo = std::max(c0 - h, p - 1.0f), hi = std::min(c0 + h, p + 1.0f);
                    return std::max(0.0f, hi - lo) / 2.0f;
                };
                truth[v] = r + C.drho * ov2(tc.x, x) * ov2(tc.y, y) * ov2(tc.z, z);
                domain[v] = (x >= -60 && x <= 60 && y >= -40 && y <= 80 && z >= -190 && z <= -10) ? 1 : 0;
            }
        });
        std::vector<float> mu0(prior.size());
        for (size_t v = 0; v < prior.size(); ++v) mu0[v] = kap * prior[v];
        VoxelMedium mg; mg.lo = kRmLo; mg.voxel = rv; mg.nx = rnx; mg.ny = rny; mg.nz = rnz;
        mg.mu_rock = kap * kRmHost; mg.mu_air = kap * kRmAir; mg.mu = mu0.data();
        const double W = 1.0e4;
        std::vector<MuonBinnedData> bd(kRmND);
        for (int dI = 0; dI < kRmND; ++dI) { bd[dI].sky = g.sky; bd[dI].open.assign(nb, 0); bd[dI].det.assign(nb, 0); }
        for (size_t i = 0; i < NB; ++i) {
            const int dI = sdet[i], b = sbin[i];
            const double No = sw[i] * T180 * kRmRatePerM2Day * double(open[b]) / g.M;
            if (No < 1.0) continue;
            const float c = bd[dI].bin_direction(b).z;
            const float Tobs = float(std::min(std::max(dEx[i], 0.5) / No, 1.0));
            const float X = ttab_invert(htt, c, Tobs);
            bd[dI].open[b] = (unsigned long long)std::llround(W * No);
            bd[dI].det[b] = (unsigned long long)std::max(1LL, std::llround(W * No * std::exp(-kap * X)));
        }
        std::vector<MuonView> views;
        for (int dI = 0; dI < kRmND; ++dI) views.push_back({kRmDet[dI], &bd[dI]});
        std::vector<float> rec = mlem_transmission(mg, views, 20, mu0, domain);
        for (float& v : rec) v /= kap;
        write_bin(out + "/recon2m_example_s1.f32", rec);
        write_bin(out + "/truth2m_example.f32", truth);
        write_bin(out + "/domain2m.u8", domain);
        std::printf("\n  MLEM density recon (cell %d, seed 1, T = 180 d, 2 m grid, 20 iters) in %.1f s\n", kRmExample, secs(t0));
    }

    FILE* fm = std::fopen((out + "/meta.txt").c_str(), "w");
    std::fprintf(fm, "log2M %d\nseeds %d\nnb %d\nn_th %d\nn_az %d\nth_max %f\nndet %d\nNB %zu\n", log2M, nSeeds, nb, g.sky.n_th,
                 g.sky.n_az, g.sky.th_max, kRmND, NB);
    std::fprintf(fm, "scan_cell lo -60 -40 -190 step 2 n 61 61 91\nscan_blind lo -70 -70 -190 step 2 n 71 71 91\n");
    std::fprintf(fm, "vol2m 240 210 102 lo -240 -210 -204 vox 2\nexample_cell %d\njv_1gev %f\njv_p0 %f\nexposures 30 45 90 180 365\n",
                 kRmExample, jv1, jvp0);
    std::fprintf(fm, "blind_err %.3f\nverdict_blind %d\nfailures %d\n", blind_err, verdict_blind, failures);
    std::fclose(fm);
    std::fclose(fx);

    cudaFree(d_bins); cudaFree(d_dF);
    cudaFree(g.d_rho); cudaFree(g.d_tsum); cudaFree(g.d_ull); cudaFree(g.d_tab); cudaFree(g.d_acc);
    std::printf("\n  total wall time %.1f s\n%s\n\n", secs(T0), failures ? "VALIDATION FAILURES PRESENT" : "validation checks passed");
    return failures ? 1 : 0;
}
