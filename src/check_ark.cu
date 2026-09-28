// =============================================================================
// QuBLAR -- check_ark: the Durupinar formation as a muography FALSIFICATION test
// =============================================================================
// ADR-010 (pre-registered; frozen hash in experiments/ark/PREREG_SHA256.txt).
// Synthetic only. Reuses the muon path of ADR-006: march_medium (the one shared
// Amanatides-Woo marcher), sample_sky (cos^2 sky, hash-keyed candidates), the
// imaging_sky() 1 x 1 deg binning, MuonBinnedData / MuonView and
// mlem_transmission for the density slices. What is new here:
//   * the medium holds DENSITY (g/cm^3), so the marched sum is opacity in m.w.e.;
//   * survival is T(X, theta) from the Reyna/Bugaev spectrum + CSDA range
//     (shallow scenes: 40-200 m.w.e.), tabulated on the host, looked up on GPU;
//   * expected maps (per-bin mean T over 2^23 hash-keyed rays) + Poisson counts,
//     cross-checked against one direct Bernoulli exposure;
//   * the ADR-010 section 5 statistic and section 6 decision rule.
// Nothing here says what is in the ground. It says what a campaign could tell.
// =============================================================================

#include "muon.cuh"
#include "muon_recon.hpp"
#include "muon_replica.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace argos;

// ---------------------------------------------------------------- ADR-010 s2
static const float kVox = 0.5f;
static const int kNX = 416, kNY = 288, kNZ = 56;
static const float3 kGridLo = make_float3(-104.0f, -72.0f, -22.0f);

static const float kMoundL = 160.0f, kMoundW = 45.0f, kRelief = 4.0f;
static const float kBodyBottom = -16.0f;
static const float kHullHalfLen = 78.5f, kHullHalfBeam = 13.0f, kHullTaper = 15.0f;
static const float kHullTop = 2.5f, kHullBot = -13.2f;
static const float kShell = 1.0f, kInner = 0.5f, kBulkheadPitch = 13.0f;

// ---------------------------------------------------------------- ADR-010 s3
static const float kRhoAir = 0.0012f, kRhoGround = 2.0f, kRhoFormMean = 2.35f;
static const float kRhoWood = 0.75f, kRhoSed = 1.9f, kRhoPetr = 2.5f;
static const float kNoise = 0.03f;
static const float3 kVoidC = make_float3(20.0f, 0.0f, -5.0f);
static const float kVoidL[6] = {1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f};

// ---------------------------------------------------------------- ADR-010 s4
static const int kNDet = 6;
static const float3 kDet[kNDet] = {
    make_float3(-45.0f, -15.0f, -20.0f), make_float3(0.0f, -15.0f, -20.0f),
    make_float3(45.0f, -15.0f, -20.0f),  make_float3(-45.0f, 15.0f, -20.0f),
    make_float3(0.0f, 15.0f, -20.0f),    make_float3(45.0f, 15.0f, -20.0f)};
static const double kDetArea = 1.0;          // m^2 per station
static const double kRatePerM2Day = 1.44e7;  // 1 cm^-2 min^-1 (PDG, sea level)
static const int kExpDays[4] = {1, 7, 30, 90};
static const int kK = 9;                     // density-scale nuisance grid
static const float kS0 = 0.92f, kDS = 0.02f;

enum Hyp { H_A = 0, H_B1, H_B2, H_B3, H_V0 };   // H_V0 + i = void kVoidL[i]
// POST-HOC, NOT PRE-REGISTERED (exploratory, labelled as such in every output):
// B3x = walls/decks at 2.5, compartments left as the surrounding natural material.
static const int H_B3X = 10;
static const int kNHyp = 4 + 6 + 1;
static const char* hyp_name(int h) {
    static const char* n[] = {"A", "B1", "B2", "B3", "V1m", "V2m", "V3m", "V4m", "V6m", "V8m", "B3x"};
    return n[h];
}

// ---------------------------------------------------------------- geometry
static float mound_halfwidth(float x) {
    const float u = x / (0.5f * kMoundL);
    if (fabsf(u) >= 1.0f) return 0.0f;
    return x < 0.0f ? 0.5f * kMoundW * sqrtf(1.0f - u * u) : 0.5f * kMoundW * (1.0f - u * u);
}
static bool in_planform(float x, float y) { return fabsf(y) < mound_halfwidth(x); }
static float surface_z(float x, float y) {
    const float w = mound_halfwidth(x);
    if (fabsf(y) >= w) return 0.0f;
    const float r = y / w;
    return kRelief * sqrtf(1.0f - r * r);
}
static float hull_halfbeam(float x) {
    return kHullHalfBeam * std::min(1.0f, (kHullHalfLen - fabsf(x)) / kHullTaper);
}
// 0 none, 1 wall/deck/bulkhead, 2 compartment
static int hull_class(float x, float y, float z) {
    if (fabsf(x) > kHullHalfLen) return 0;
    const float b = hull_halfbeam(x);
    if (b <= 0.0f || fabsf(y) > b) return 0;
    const float zc = std::min(kHullTop, surface_z(x, y) - 1.0f);
    if (z < kHullBot || z > zc) return 0;
    if (b - fabsf(y) < kShell || z - kHullBot < kShell || zc - z < kShell) return 1;
    const float storey = (kHullTop - kHullBot) / 3.0f;
    for (int d = 1; d <= 2; ++d)
        if (fabsf(z - (kHullBot + d * storey)) < 0.5f * kInner) return 1;
    const float xb = fmodf(x + kHullHalfLen, kBulkheadPitch);
    if (xb < 0.5f * kInner || kBulkheadPitch - xb < 0.5f * kInner) return 1;
    if (fabsf(fabsf(y) - 2.0f) < 0.5f * kInner) return 1;
    return 2;
}

struct ArkScene {
    int hyp = H_A;
    int seed = -1;                 // -1: ensemble-mean model
    std::vector<float> lay_top;    // layer upper boundaries in zeta
    std::vector<float> lay_rho;
};

static ArkScene make_scene(int hyp, int seed) {
    ArkScene s;
    s.hyp = hyp;
    s.seed = seed;
    if (seed >= 0) {
        std::mt19937_64 rng(0xA5C0FFEEull ^ (unsigned long long)(seed) * 0x9E3779B97F4A7C15ull);
        std::uniform_real_distribution<float> th(0.5f, 2.0f), rh(2.0f, 2.7f);
        float z = -30.0f;
        while (z < 8.0f) {
            z += th(rng);
            s.lay_top.push_back(z);
            s.lay_rho.push_back(rh(rng));
        }
    }
    return s;
}

static float layer_rho(const ArkScene& s, float zeta) {
    auto it = std::lower_bound(s.lay_top.begin(), s.lay_top.end(), zeta);
    if (it == s.lay_top.end()) return s.lay_rho.back();
    return s.lay_rho[it - s.lay_top.begin()];
}

static float gauss_hash(unsigned key) {
    const float u1 = 1.0f - hash_unit(key);
    const float u2 = hash_unit(key ^ 0x5bd1e995u);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

static float rho_at(const ArkScene& s, float x, float y, float z, unsigned key) {
    if (z > surface_z(x, y)) return kRhoAir;
    const bool body = in_planform(x, y) && z >= kBodyBottom;
    float r = kRhoGround;
    if (body) {
        if (s.seed < 0) r = kRhoFormMean;
        else {
            const float zeta = z - 6.0f * (y / 22.5f) * (y / 22.5f) - 6.0f * (x / 80.0f) * (x / 80.0f);
            r = layer_rho(s, zeta);
        }
    }
    if (s.hyp >= H_B1 && s.hyp <= H_B3) {
        const int hc = hull_class(x, y, z);
        if (hc == 1) r = (s.hyp == H_B3) ? kRhoPetr : kRhoWood;
        else if (hc == 2) r = (s.hyp == H_B1) ? kRhoAir : kRhoSed;
    }
    if (s.hyp == H_B3X && hull_class(x, y, z) == 1) r = kRhoPetr;
    if (s.hyp >= H_V0 && s.hyp < H_V0 + 6) {
        const float h = 0.5f * kVoidL[s.hyp - H_V0];
        if (fabsf(x - kVoidC.x) <= h && fabsf(y - kVoidC.y) <= h && fabsf(z - kVoidC.z) <= h)
            r = kRhoAir;
    }
    if (s.seed >= 0 && r > 0.01f)
        r *= 1.0f + kNoise * gauss_hash(key ^ (unsigned)(s.seed) * 0x27d4eb2du);
    return r;
}

static void parallel_for(int n, const std::function<void(int)>& f) {
    const int nt = std::max(1, std::min(16, (int)std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] { for (int i = t; i < n; i += nt) f(i); });
    for (auto& x : th) x.join();
}

static void build_volume(const ArkScene& s, float3 lo, float vox, int nx, int ny, int nz,
                         std::vector<float>& rho) {
    rho.assign(size_t(nx) * ny * nz, 0.0f);
    parallel_for(nz, [&](int k) {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const size_t idx = (size_t(k) * ny + j) * nx + i;
                rho[idx] = rho_at(s, lo.x + (i + 0.5f) * vox, lo.y + (j + 0.5f) * vox,
                                  lo.z + (k + 0.5f) * vox, (unsigned)idx * 0x9E3779B1u);
            }
    });
}

// ---------------------------------------------------------------- muon physics
// Reyna (2006) sea-level spectrum, I(p, theta) = cos^3 theta I_V(p cos theta).
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

static std::vector<float> build_ttable(int nc, int nx, float c0, float c1, float xmax,
                                       double& jv_1gev, double& jv_p0) {
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
            J[i] = J[i + 1] + 0.5 * (f0 + f1) * (lp1 - lp0) / (np - 1);   // dp = p dlnp
        }
        auto Jat = [&](double pm) {
            if (pm <= p[0]) return J[0];
            if (pm >= p[np - 1]) return 0.0;
            const double f = (std::log(pm) - lp0) / (lp1 - lp0) * (np - 1);
            const int i = (int)f;
            const double a = f - i;
            return (1 - a) * J[i] + a * J[i + 1];
        };
        if (ic == nc - 1) { jv_1gev = Jat(1.0) * 1e4; jv_p0 = J[0] * 1e4; }   // m^-2 s^-1 sr^-1
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
    // T is monotone decreasing in X: bisection
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

// Expected map: per bin, the sum over the candidates in it of T(c, s_k X).
__global__ void ark_expected(const VoxelMedium m, MuonSky sky, int n, float3 chamber,
                             TTable tt, int K, float s0, float ds,
                             double* tsum, unsigned long long* open) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned key = hash_u32(static_cast<unsigned>(i) * 0x9E3779B9u);   // as ADR-006
    float3 d; float c, az;
    sample_sky(false, sky.c_min, key, d, c, az);
    const int bin = sky_bin(sky, c, az);
    if (open) atomicAdd(&open[bin], 1ULL);
    const float X = march_medium(m, chamber, d);   // density grid -> opacity, m.w.e.
    for (int k = 0; k < K; ++k)
        atomicAdd(&tsum[size_t(k) * sky.size() + bin], (double)ttab(tt, c, (s0 + k * ds) * X));
}

// Direct Bernoulli exposure, the validation of the expected-map shortcut.
__global__ void ark_bernoulli(const VoxelMedium m, MuonSky sky, int n, float3 chamber,
                              TTable tt, unsigned seed, unsigned long long* det) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned key = hash_u32(static_cast<unsigned>(i) * 0x9E3779B9u);
    float3 d; float c, az;
    sample_sky(false, sky.c_min, key, d, c, az);
    const int bin = sky_bin(sky, c, az);
    const float X = march_medium(m, chamber, d);
    if (hash_unit(key ^ 0x2545F491u ^ seed) < ttab(tt, c, X)) atomicAdd(&det[bin], 1ULL);
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
    float* d_tab = nullptr;
    TTable tt{};
    MuonSky sky;
    int M = 1 << 23;
};

static VoxelMedium sim_medium(const Gpu& g) {
    VoxelMedium m;
    m.lo = kGridLo; m.voxel = kVox; m.nx = kNX; m.ny = kNY; m.nz = kNZ;
    m.mu_rock = kRhoFormMean; m.mu_air = kRhoAir; m.mu = g.d_rho;
    return m;
}

// Upload a scene, run all detectors, return tsum[det][k][bin] (host doubles).
static std::vector<double> run_scene(Gpu& g, const ArkScene& s, int K,
                                     std::vector<unsigned long long>* open_out,
                                     std::vector<float>* keep_rho = nullptr) {
    std::vector<float> rho;
    build_volume(s, kGridLo, kVox, kNX, kNY, kNZ, rho);
    CUDA_CHECK(cudaMemcpy(g.d_rho, rho.data(), rho.size() * sizeof(float), cudaMemcpyHostToDevice));
    if (keep_rho) *keep_rho = rho;
    const int nb = g.sky.size();
    std::vector<double> out(size_t(kNDet) * K * nb);
    const VoxelMedium m = sim_medium(g);
    for (int dI = 0; dI < kNDet; ++dI) {
        CUDA_CHECK(cudaMemset(g.d_tsum, 0, size_t(K) * nb * sizeof(double)));
        if (open_out) CUDA_CHECK(cudaMemset(g.d_ull, 0, nb * sizeof(unsigned long long)));
        ark_expected<<<muon_grid(g.M, 256), 256>>>(m, g.sky, g.M, kDet[dI], g.tt, K,
                                                    K == 1 ? 1.0f : kS0, kDS,
                                                    g.d_tsum, open_out ? g.d_ull : nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data() + size_t(dI) * K * nb, g.d_tsum,
                              size_t(K) * nb * sizeof(double), cudaMemcpyDeviceToHost));
        if (open_out && dI == 0) {
            open_out->resize(nb);
            CUDA_CHECK(cudaMemcpy(open_out->data(), g.d_ull, nb * sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost));
        }
    }
    return out;
}

// F(d | m) = sum_b m_b - d_b ln m_b (deviance/2 up to a data-only constant)
static double F_eval(const std::vector<double>& d, const double* mu, const double* lnmu,
                     double sum_mu, double E) {
    double s = 0.0;
    for (size_t b = 0; b < d.size(); ++b) s += d[b] * lnmu[b];
    return E * sum_mu - s;   // the -ln(E) sum_b d_b term is common to all models
}
static double profile_min(const double* F) {
    int k = 0;
    for (int i = 1; i < kK; ++i) if (F[i] < F[k]) k = i;
    if (k == 0 || k == kK - 1) return F[k];
    const double a = F[k - 1], b = F[k], c = F[k + 1];
    const double curv = a - 2 * b + c;
    if (curv <= 0) return b;
    return b - (c - a) * (c - a) / (8.0 * curv);
}

int main(int argc, char** argv) {
    const std::string out = argc > 1 ? argv[1] : "experiments/ark";
    const int log2M = argc > 2 ? std::atoi(argv[2]) : 23;
    const int nSeeds = argc > 3 ? std::atoi(argv[3]) : 8;
    const bool do_recon = argc > 4 ? std::atoi(argv[4]) != 0 : true;
    auto T0 = std::chrono::steady_clock::now();

    std::printf("\nQuBLAR -- check_ark (ADR-010 pre-registered falsification test; SYNTHETIC)\n\n");
    std::printf("  grid %dx%dx%d @ %.1f m; %d detectors x %.0f m^2 at z = %.0f m\n", kNX, kNY, kNZ,
                kVox, kNDet, kDetArea, kDet[0].z);
    std::printf("  candidates per detector per scene 2^%d; seeds 1..%d; exposures 1/7/30/90 d\n",
                log2M, nSeeds);

    Gpu g;
    g.M = 1 << log2M;
    g.sky = imaging_sky();
    const int nb = g.sky.size();

    // transmission table
    double jv1 = 0, jvp0 = 0;
    const int NC = 128, NXT = 2048;
    const float C0 = 0.30f, C1 = 1.0f, XMAX = 600.0f;
    std::vector<float> tab = build_ttable(NC, NXT, C0, C1, XMAX, jv1, jvp0);
    CUDA_CHECK(cudaMalloc(&g.d_tab, tab.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g.d_tab, tab.data(), tab.size() * sizeof(float), cudaMemcpyHostToDevice));
    g.tt = TTable{g.d_tab, NC, NXT, C0, C1, XMAX};
    TTable htt{tab.data(), NC, NXT, C0, C1, XMAX};
    std::printf("\n  flux model (Reyna 2006): vertical J(>1 GeV) = %.1f m^-2 s^-1 sr^-1 (PDG ~70); "
                "J(>%.1f GeV/c) = %.1f\n", jv1, kP0, jvp0);
    std::printf("  vertical survival T(X): 10 mwe %.3f | 40 mwe %.3f | 60 mwe %.3f | 100 mwe %.4f | 200 mwe %.5f\n",
                ttab(htt, 1, 10), ttab(htt, 1, 40), ttab(htt, 1, 60), ttab(htt, 1, 100), ttab(htt, 1, 200));

    CUDA_CHECK(cudaMalloc(&g.d_rho, size_t(kNX) * kNY * kNZ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_tsum, size_t(kK) * nb * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.d_ull, nb * sizeof(unsigned long long)));

    // grid-margin check: bin-centre rays must not leave the grid below grade
    MuonBinnedData geom;
    geom.sky = g.sky;
    int leaks = 0;
    for (int dI = 0; dI < kNDet; ++dI)
        for (int b = 0; b < nb; ++b) {
            const float3 d = geom.bin_direction(b), p = kDet[dI];
            float t = 1e30f;
            const float hi[3] = {kGridLo.x + kNX * kVox, kGridLo.y + kNY * kVox, kGridLo.z + kNZ * kVox};
            const float lo[3] = {kGridLo.x, kGridLo.y, kGridLo.z};
            const float pc[3] = {p.x, p.y, p.z}, dc[3] = {d.x, d.y, d.z};
            for (int a = 0; a < 3; ++a) {
                if (fabsf(dc[a]) < 1e-9f) continue;
                t = std::min(t, ((dc[a] > 0 ? hi[a] : lo[a]) - pc[a]) / dc[a]);
            }
            if (p.z + d.z * t < -0.01f) ++leaks;
        }
    std::printf("  grid margins: bin-centre rays leaving the grid below grade = %d (must be 0)\n", leaks);
    int failures = leaks ? 1 : 0;

    // ---------------- models (ensemble means, K scales) --------------------------
    auto t0 = std::chrono::steady_clock::now();
    std::vector<unsigned long long> open;
    std::vector<std::vector<double>> model(kNHyp);   // [det][k][bin] per-day expected counts
    std::vector<float> rhoA_mean;
    for (int h = 0; h < kNHyp; ++h) {
        model[h] = run_scene(g, make_scene(h, -1), kK, h == 0 ? &open : nullptr,
                             h == 0 ? &rhoA_mean : nullptr);
        for (double& v : model[h]) v = std::max(v * kRatePerM2Day * kDetArea / g.M, 1e-12);
    }
    std::printf("  models: %d scenes x %d detectors x %d scales in %.1f s\n", kNHyp, kNDet, kK, secs(t0));

    // ---------------- data realisations ------------------------------------------
    t0 = std::chrono::steady_clock::now();
    // data[h][seed] -> [det][bin] per-day expected counts of that realisation
    std::vector<std::vector<std::vector<double>>> data(kNHyp, std::vector<std::vector<double>>(nSeeds));
    for (int k = 0; k < nSeeds; ++k)
        for (int h = 0; h < kNHyp; ++h) {
            std::vector<float> keep;
            const bool dump = (k == 0 && (h <= H_B3 || h == H_B3X));
            data[h][k] = run_scene(g, make_scene(h, k + 1), 1, nullptr, dump ? &keep : nullptr);
            for (double& v : data[h][k]) v *= kRatePerM2Day * kDetArea / g.M;
            if (dump) {
                // 2 m render volume and the per-bin maps for the figures
                std::vector<float> v2;
                build_volume(make_scene(h, 1), kGridLo, 2.0f, kNX / 4, kNY / 4, kNZ / 4, v2);
                write_bin(out + "/vol2m_" + hyp_name(h) + ".f32", v2);
                std::vector<float> tm(size_t(kNDet) * nb), mm(size_t(kNDet) * nb);
                for (int dI = 0; dI < kNDet; ++dI)
                    for (int b = 0; b < nb; ++b) {
                        tm[size_t(dI) * nb + b] = float(data[h][k][size_t(dI) * nb + b]);
                        mm[size_t(dI) * nb + b] = float(model[h][(size_t(dI) * kK + 4) * nb + b]);
                    }
                write_bin(out + "/rate_real_s1_" + std::string(hyp_name(h)) + ".f32", tm);
                write_bin(out + "/rate_model_" + std::string(hyp_name(h)) + ".f32", mm);
            }
        }
    {
        std::vector<float> of(nb);
        for (int b = 0; b < nb; ++b) of[b] = float(double(open[b]) / g.M);
        write_bin(out + "/open_fraction.f32", of);
    }
    std::printf("  realisations: %d hypotheses x %d seeds in %.1f s\n", kNHyp, nSeeds, secs(t0));

    // ---------------- validation: direct Bernoulli vs expected map ----------------
    {
        const int Mb = 1 << 24;
        std::vector<float> rho;
        build_volume(make_scene(H_A, 1), kGridLo, kVox, kNX, kNY, kNZ, rho);
        CUDA_CHECK(cudaMemcpy(g.d_rho, rho.data(), rho.size() * sizeof(float), cudaMemcpyHostToDevice));
        const VoxelMedium m = sim_medium(g);
        CUDA_CHECK(cudaMemset(g.d_tsum, 0, nb * sizeof(double)));
        ark_expected<<<muon_grid(Mb, 256), 256>>>(m, g.sky, Mb, kDet[0], g.tt, 1, 1.0f, 0.0f, g.d_tsum, nullptr);
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> ex(nb);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(ex.data(), g.d_tsum, nb * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(g.d_ull, 0, nb * sizeof(unsigned long long)));
        ark_bernoulli<<<muon_grid(Mb, 256), 256>>>(m, g.sky, Mb, kDet[0], g.tt, 777u, g.d_ull);
        CUDA_CHECK(cudaGetLastError());
        std::vector<unsigned long long> det(nb);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(det.data(), g.d_ull, nb * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        double chi2 = 0; int ndf = 0; double se = 0, sd = 0;
        for (int b = 0; b < nb; ++b) {
            se += ex[b]; sd += double(det[b]);
            if (ex[b] < 5.0) continue;
            chi2 += (det[b] - ex[b]) * (det[b] - ex[b]) / ex[b];
            ++ndf;
        }
        const bool ok = ndf > 0 && chi2 / ndf > 0.8 && chi2 / ndf < 1.2;
        std::printf("  validation (H_A seed 1, det 0, 2^24 Bernoulli, seed 777): expected %.0f, detected %.0f, "
                    "chi2/ndf = %.3f (ndf %d) %s\n", se, sd, chi2 / ndf, ndf, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // ---------------- statistic ---------------------------------------------------
    t0 = std::chrono::steady_clock::now();
    const size_t NB = size_t(kNDet) * nb;
    // per model and scale: contiguous mu and ln mu over all detectors' bins
    std::vector<std::vector<double>> mu(kNHyp * kK, std::vector<double>(NB)), lnmu(kNHyp * kK, std::vector<double>(NB));
    std::vector<double> summu(kNHyp * kK, 0.0);
    for (int h = 0; h < kNHyp; ++h)
        for (int k = 0; k < kK; ++k) {
            auto& M_ = mu[h * kK + k]; auto& L_ = lnmu[h * kK + k];
            for (int dI = 0; dI < kNDet; ++dI)
                for (int b = 0; b < nb; ++b) {
                    const double v = model[h][(size_t(dI) * kK + k) * nb + b];
                    M_[size_t(dI) * nb + b] = v;
                    L_[size_t(dI) * nb + b] = std::log(v);
                }
            summu[h * kK + k] = std::accumulate(M_.begin(), M_.end(), 0.0);
        }
    // q[pair X][E][seed] under A-data and under X-data
    const int nPair = kNHyp - 1;
    std::vector<double> qA(size_t(nPair) * 4 * nSeeds), qX(size_t(nPair) * 4 * nSeeds);
    // Profile over s in [0.92, 1.08]: grid, then golden section on a cubic-Hermite
    // interpolation of ln m between the grid points (the 3-point parabola of the
    // first draft was not accurate enough when the curvature in s dwarfs the
    // residual: it produced Z_nuis = 0 for a 1 m void, which is impossible).
    auto Fs = [&](const std::vector<double>& d, int h, double E, double sv) {
        double f = (sv - kS0) / kDS;
        int k = std::min(kK - 2, std::max(0, (int)std::floor(f)));
        const double t = std::min(1.0, std::max(0.0, f - k));
        const double h00 = 2*t*t*t - 3*t*t + 1, h10 = t*t*t - 2*t*t + t, h01 = -2*t*t*t + 3*t*t, h11 = t*t*t - t*t;
        const double* L0 = lnmu[h * kK + k].data();
        const double* L1 = lnmu[h * kK + k + 1].data();
        const double* Lm = lnmu[h * kK + std::max(0, k - 1)].data();
        const double* Lp = lnmu[h * kK + std::min(kK - 1, k + 2)].data();
        const double w0 = (k == 0) ? 1.0 : 0.5, w1 = (k + 1 == kK - 1) ? 1.0 : 0.5;
        double sm = 0, sd = 0;
        for (size_t b = 0; b < d.size(); ++b) {
            const double m0 = w0 * (L1[b] - Lm[b]), m1 = w1 * (Lp[b] - L0[b]);
            const double L = h00 * L0[b] + h10 * m0 + h01 * L1[b] + h11 * m1;
            sm += std::exp(L);
            sd += d[b] * L;
        }
        return E * sm - sd;
    };
    auto Fmin = [&](const std::vector<double>& d, int h, double E) {
        double F[kK];
        for (int k = 0; k < kK; ++k)
            F[k] = F_eval(d, mu[h * kK + k].data(), lnmu[h * kK + k].data(), summu[h * kK + k], E);
        int km = 0;
        for (int k = 1; k < kK; ++k) if (F[k] < F[km]) km = k;
        double a = kS0 + kDS * std::max(0, km - 1), c = kS0 + kDS * std::min(kK - 1, km + 1);
        const double gr = 0.6180339887498949;
        double x1 = c - gr * (c - a), x2 = a + gr * (c - a);
        double f1 = Fs(d, h, E, x1), f2 = Fs(d, h, E, x2);
        for (int it = 0; it < 30; ++it) {
            if (f1 < f2) { c = x2; x2 = x1; f2 = f1; x1 = c - gr * (c - a); f1 = Fs(d, h, E, x1); }
            else { a = x1; x1 = x2; f1 = f2; x2 = a + gr * (c - a); f2 = Fs(d, h, E, x2); }
        }
        return std::min(F[km], std::min(f1, f2));
    };
    parallel_for(nSeeds, [&](int k) {
        for (int e = 0; e < 4; ++e) {
            const double E = kExpDays[e];
            auto draw = [&](int h, unsigned long long tag) {
                std::mt19937_64 rng(tag);
                std::vector<double> d(NB);
                for (size_t b = 0; b < NB; ++b) {
                    const double lam = data[h][k][b] * E;
                    d[b] = lam > 0 ? double(std::poisson_distribution<long long>(lam)(rng)) : 0.0;
                }
                return d;
            };
            const unsigned long long base = 0x5EEDull * 1000003ull + (k + 1) * 1009ull + e * 17ull;
            const std::vector<double> dA = draw(H_A, base * 31 + 0);
            const double FA_A = Fmin(dA, H_A, E);
            for (int x = 1; x < kNHyp; ++x) {
                const size_t o = (size_t(x - 1) * 4 + e) * nSeeds + k;
                qA[o] = 2.0 * (FA_A - Fmin(dA, x, E));
                const std::vector<double> dX = draw(x, base * 31 + x);
                qX[o] = 2.0 * (Fmin(dX, H_A, E) - Fmin(dX, x, E));
            }
        }
    });
    std::printf("  statistic: %d pairs x 4 exposures x %d seeds in %.1f s\n\n", nPair, nSeeds, secs(t0));

    // ---------------- summary + decision -----------------------------------------
    FILE* fq = std::fopen((out + "/q_values.csv").c_str(), "w");
    FILE* fs = std::fopen((out + "/summary.csv").c_str(), "w");
    std::fprintf(fq, "pair,exposure_days,seed,q_under_A,q_under_X\n");
    std::fprintf(fs, "pair,exposure_days,Z_emp,AUC,median_qX,mean_qA,sd_qA,Z_asimov_ideal,Z_asimov_nuis\n");
    std::printf("  %-7s %5s %10s %6s %12s %12s %10s %9s %9s\n", "pair", "days", "Z_emp", "AUC",
                "med q_X", "mean q_A", "sd q_A", "Z_ideal", "Z_nuis");
    std::vector<int> verdict(nPair, 0);
    std::vector<double> z30(nPair), z90(nPair);
    for (int x = 1; x < kNHyp; ++x) {
        // Asimov per day, scaled by E
        double D1 = 0; double Dn[kK];
        const auto& mX = mu[x * kK + 4];
        for (size_t b = 0; b < NB; ++b) { const double a = mu[H_A * kK + 4][b]; D1 += 2 * (a - mX[b] + mX[b] * std::log(mX[b] / a)); }
        for (int s = 0; s < kK; ++s) {
            double D = 0;
            for (size_t b = 0; b < NB; ++b) { const double a = mu[H_A * kK + s][b]; D += 2 * (a - mX[b] + mX[b] * std::log(mX[b] / a)); }
            Dn[s] = D;
        }
        (void)Dn;
        double Fsat = 0;
        for (size_t b = 0; b < NB; ++b) Fsat += mX[b] - mX[b] * std::log(mX[b]);
        const double Dn1 = 2.0 * (Fmin(mX, H_A, 1.0) - Fsat);
        for (int e = 0; e < 4; ++e) {
            const double E = kExpDays[e];
            std::vector<double> a(nSeeds), xx(nSeeds);
            for (int k = 0; k < nSeeds; ++k) {
                const size_t o = (size_t(x - 1) * 4 + e) * nSeeds + k;
                a[k] = qA[o]; xx[k] = qX[o];
                std::fprintf(fq, "A_vs_%s,%d,%d,%.6g,%.6g\n", hyp_name(x), kExpDays[e], k + 1, a[k], xx[k]);
            }
            const double mA = std::accumulate(a.begin(), a.end(), 0.0) / nSeeds;
            double vA = 0; for (double v : a) vA += (v - mA) * (v - mA);
            const double sA = std::sqrt(vA / std::max(1, nSeeds - 1));
            std::vector<double> sx = xx; std::sort(sx.begin(), sx.end());
            const double med = nSeeds % 2 ? sx[nSeeds / 2] : 0.5 * (sx[nSeeds / 2 - 1] + sx[nSeeds / 2]);
            double auc = 0; for (double u : xx) for (double v : a) auc += (u > v) ? 1.0 : (u == v ? 0.5 : 0.0);
            auc /= double(nSeeds) * nSeeds;
            const double Z = sA > 0 ? (med - mA) / sA : 0.0;
            const double Zi = std::sqrt(std::max(0.0, E * D1)), Zn = std::sqrt(std::max(0.0, E * Dn1));
            std::fprintf(fs, "A_vs_%s,%d,%.4g,%.4g,%.6g,%.6g,%.6g,%.4g,%.4g\n", hyp_name(x), kExpDays[e], Z, auc,
                         med, mA, sA, Zi, Zn);
            std::printf("  A-%-5s %5d %10.2f %6.3f %12.1f %12.1f %10.1f %9.1f %9.1f\n", hyp_name(x), kExpDays[e],
                        Z, auc, med, mA, sA, Zi, Zn);
            if (kExpDays[e] <= 30 && Z >= 5.0 && auc >= 0.99) verdict[x - 1] = 1;
            if (kExpDays[e] == 30) z30[x - 1] = Z;
            if (kExpDays[e] == 90) z90[x - 1] = Z;
        }
    }
    std::fclose(fq);
    std::fclose(fs);

    std::printf("\n  DECISION (ADR-010 s6: Z >= 5 and AUC >= 0.99 at <= 30 d, 6 m^2):\n");
    for (int x : {1, 2, 3, H_B3X})
        std::printf("    A vs %-3s : %s  (Z_emp 30 d = %.2f, 90 d = %.2f)\n", hyp_name(x),
                    verdict[x - 1] ? "DISTINGUISHABLE" : "NOT distinguishable at <= 30 d", z30[x - 1], z90[x - 1]);
    std::printf("    (B3x is POST-HOC / exploratory, not part of the pre-registered decision)\n");
    float min_void = -1;
    for (int i = 0; i < 6; ++i)
        if (verdict[H_V0 + i - 1]) { min_void = kVoidL[i]; break; }
    for (int i = 0; i < 6; ++i)
        std::printf("    A vs void %.0f m cube: %s (Z_emp 30 d = %.2f)\n", kVoidL[i],
                    verdict[H_V0 + i - 1] ? "detectable" : "not detectable", z30[H_V0 + i - 1]);
    if (min_void > 0) std::printf("    minimum detectable void (<= 30 d): %.0f m cube\n", min_void);
    else std::printf("    minimum detectable void (<= 30 d): none of the swept sizes (max 8 m)\n");

    // ---------------- reconstruction (figures; not part of the decision) ----------
    if (do_recon) {
        t0 = std::chrono::steady_clock::now();
        const float rv = 1.0f; const int rnx = 208, rny = 144, rnz = 28;
        const float kap = 0.05f;
        std::vector<float> prior;
        build_volume(make_scene(H_A, -1), kGridLo, rv, rnx, rny, rnz, prior);
        std::vector<char> domain(prior.size(), 0);
        for (int kz = 0; kz < rnz; ++kz) for (int j = 0; j < rny; ++j) for (int i = 0; i < rnx; ++i) {
            const float x = kGridLo.x + (i + 0.5f) * rv, y = kGridLo.y + (j + 0.5f) * rv, z = kGridLo.z + (kz + 0.5f) * rv;
            domain[(size_t(kz) * rny + j) * rnx + i] = (in_planform(x, y) && z >= kBodyBottom && z <= surface_z(x, y)) ? 1 : 0;
        }
        std::vector<float> mu0(prior.size());
        for (size_t v = 0; v < prior.size(); ++v) mu0[v] = kap * prior[v];
        VoxelMedium mg; mg.lo = kGridLo; mg.voxel = rv; mg.nx = rnx; mg.ny = rny; mg.nz = rnz;
        mg.mu_rock = kap * kRhoFormMean; mg.mu_air = kap * kRhoAir; mg.mu = mu0.data();
        const double E = 30.0, W = 1.0e4;
        for (int h = 0; h <= H_B3X; ++h) {
            if (h > H_B3 && h != H_B3X) continue;
            std::vector<MuonBinnedData> bd(kNDet);
            std::mt19937_64 rng(0xBEEFull + h);
            for (int dI = 0; dI < kNDet; ++dI) {
                bd[dI].sky = g.sky; bd[dI].open.resize(nb); bd[dI].det.resize(nb);
                for (int b = 0; b < nb; ++b) {
                    const double No = E * kRatePerM2Day * kDetArea * double(open[b]) / g.M;
                    const double lam = data[h][0][size_t(dI) * nb + b] * E;
                    const double dcount = lam > 0 ? double(std::poisson_distribution<long long>(lam)(rng)) : 0.0;
                    if (No < 1.0) { bd[dI].open[b] = 0; bd[dI].det[b] = 0; continue; }
                    const float c = bd[dI].bin_direction(b).z;
                    const float Tobs = float(std::min(std::max(dcount, 0.5) / No, 1.0));
                    const float X = ttab_invert(htt, c, Tobs);
                    bd[dI].open[b] = (unsigned long long)std::llround(W * No);
                    bd[dI].det[b] = (unsigned long long)std::max(1LL, std::llround(W * No * std::exp(-kap * X)));
                }
            }
            std::vector<MuonView> views;
            for (int dI = 0; dI < kNDet; ++dI) views.push_back({kDet[dI], &bd[dI]});
            std::vector<float> rec = mlem_transmission(mg, views, 30, mu0, domain);
            for (float& v : rec) v /= kap;
            write_bin(out + "/recon1m_" + std::string(hyp_name(h)) + ".f32", rec);
            std::vector<float> tr;
            build_volume(make_scene(h, 1), kGridLo, rv, rnx, rny, rnz, tr);
            write_bin(out + "/truth1m_" + std::string(hyp_name(h)) + ".f32", tr);
        }
        write_bin(out + "/domain1m.u8", domain);
        std::printf("\n  MLEM density recon (seed 1, 30 d, 1 m grid, 30 iters) in %.1f s\n", secs(t0));
    }

    FILE* fm = std::fopen((out + "/meta.txt").c_str(), "w");
    std::fprintf(fm, "log2M %d\nseeds %d\nnb %d\nn_th %d\nn_az %d\nth_max %f\nndet %d\n", log2M, nSeeds, nb,
                 g.sky.n_th, g.sky.n_az, g.sky.th_max, kNDet);
    std::fprintf(fm, "vol2m 104 72 14 lo -104 -72 -22 vox 2\nvol1m 208 144 28 lo -104 -72 -22 vox 1\n");
    std::fprintf(fm, "jv_1gev %f\njv_p0 %f\nleaks %d\n", jv1, jvp0, leaks);
    std::fclose(fm);

    cudaFree(g.d_rho); cudaFree(g.d_tsum); cudaFree(g.d_ull); cudaFree(g.d_tab);
    std::printf("\n  total wall time %.1f s\n%s\n\n", secs(T0), failures ? "VALIDATION FAILURES PRESENT" : "validation checks passed");
    return failures ? 1 : 0;
}
