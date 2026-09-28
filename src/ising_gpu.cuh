// =============================================================================
// QuBLAR -- the integer path on the GPU (ADR-016)
// =============================================================================
// The P1 engine argos::wising::anneal (sequential Metropolis sweep, splitmix64
// draw only when dE > 0, integer fields and energies), run as one warp per
// replica and many replicas per launch.
//
// Inside a replica the warp looks at 32 consecutive spins at once: every lane
// computes dE on the current fields, takes the draw it would get in sequential
// order (splitmix64 is counter-based: draw k = mix(s0 + (k+1)*gamma)), and the
// first accepting lane L is flipped; the window restarts at L+1. Spins before L
// had dE > 0 and were rejected on unchanged fields, so the trajectory is the
// sequential one -- bit-identical to the CPU engine for the same seed.
//
// Acceptance: a float fast path whose error (< 2e-5 relative for |x| <= 40) is
// far inside its 1e-3 margin, and the P1 double expression when the draw lands
// inside the margin. The fast path therefore never changes a decision.
//
// A batch is split into launches of a few sweeps (Windows WDDM watchdog); the
// state (fields, spins, draw counter, energy) is carried in global memory
// between launches, so the result does not depend on the split.
// =============================================================================

#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ising_weighted.hpp"

namespace argos {
namespace gising {

using wising::Budget;
using wising::WeightedGraph;

#define GISING_CUDA(x)                                                                        \
    do {                                                                                      \
        cudaError_t e_ = (x);                                                                 \
        if (e_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, \
                         __LINE__);                                                           \
            std::exit(3);                                                                     \
        }                                                                                     \
    } while (0)

constexpr int kWarpsPerBlock = 4;
constexpr int kSharedStateLimit = 16384;   // bytes of replica state per warp (ADR-016 s2)

// ---- the plan: types, storage, state placement, temperatures ----------------------

struct Plan {
    int n = 0;
    int64_t m = 0;
    int n_pad = 0;
    int64_t B = 0;           // max_i sum_j |w_ij|: the field bound
    int64_t max_w = 0;       // max |w|
    int field_bytes = 4;     // 1, 2 or 4
    int weight_bytes = 4;    // 1 or 4
    bool dense = false;
    bool shared_state = false;
    double sigma = 0.0;      // sqrt((2/n) sum w^2)
    int64_t w_min = 0;       // min |w| over w != 0
    std::string refused;
};

inline Plan make_plan(const WeightedGraph& g, bool force_global = false) {
    Plan p;
    p.n = g.n;
    p.m = g.m;
    p.n_pad = (g.n + 15) / 16 * 16;
    double sum_w2 = 0.0;
    p.w_min = 0;
    for (int64_t e = 0; e < g.m; ++e) {
        const int64_t a = g.ew[size_t(e)] < 0 ? -int64_t(g.ew[size_t(e)]) : int64_t(g.ew[size_t(e)]);
        sum_w2 += double(a) * double(a);
        if (a > p.max_w) p.max_w = a;
        if (a != 0 && (p.w_min == 0 || a < p.w_min)) p.w_min = a;
    }
    for (int i = 0; i < g.n; ++i) {
        int64_t acc = 0;
        for (int64_t k = g.row[size_t(i)]; k < g.row[size_t(i) + 1]; ++k)
            acc += g.w[size_t(k)] < 0 ? -int64_t(g.w[size_t(k)]) : int64_t(g.w[size_t(k)]);
        if (acc > p.B) p.B = acc;
    }
    if (p.B > 2147483647LL) p.refused = "field bound exceeds int32";
    if (p.w_min == 0) p.refused = "all weights are zero";
    p.field_bytes = p.B <= 127 ? 1 : (p.B <= 32767 ? 2 : 4);
    p.weight_bytes = p.max_w <= 127 ? 1 : 4;
    p.dense = g.n <= 16384 && double(2 * g.m) / double(g.n) >= double(g.n) / 4.0;
    p.shared_state = !force_global && int64_t(g.n) * (p.field_bytes + 1) <= kSharedStateLimit;
    p.sigma = std::sqrt(2.0 * sum_w2 / double(g.n));
    return p;
}

/// ADR-016 section 4: T_hot = sigma, T_cold = 0.1 * w_min, P1's geometric form.
inline Budget scaled_budget(const Plan& p, int anneal, int hold) {
    Budget b;
    b.t_hot = p.sigma;
    b.t_cold = 0.1 * double(p.w_min);
    b.anneal_sweeps = anneal;
    b.hold_sweeps = hold;
    return b;
}

/// The temperature of every sweep, by the expression of argos::wising::anneal.
inline std::vector<double> temperatures(const Budget& b) {
    const int total = b.anneal_sweeps + b.hold_sweeps;
    std::vector<double> t(size_t(total), 0.0);
    for (int sweep = 0; sweep < total; ++sweep) {
        const double frac = std::min(1.0, double(sweep) / std::max(1, b.anneal_sweeps - 1));
        volatile double temp = sweep < b.anneal_sweeps
            ? b.t_hot * std::pow(b.t_cold / b.t_hot, frac) : b.t_cold;
        t[size_t(sweep)] = temp;
    }
    return t;
}

// ---- device code ----------------------------------------------------------------

namespace dev {

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// u = u53 * 2^-53; decides u < exp(-dE/T) exactly as the P1 host expression.
__device__ __forceinline__ bool metropolis(uint64_t u53, int dE, double T, float inv_t) {
    if (u53 != 0) {
        const float xf = __fmul_rn(-float(dE), inv_t);
        if (xf < -40.0f) return false;                 // exp(x) < 2^-53 <= u
        const float y = __expf(xf);
        const float uf = __fmul_rn(__ull2float_rn(u53), 0x1p-53f);
        if (uf < __fmul_rn(y, 0.999f)) return true;
        if (uf > __fmul_rn(y, 1.001f)) return false;
    }
    return double(u53) * 0x1p-53 < exp(-double(dE) / T);
}

struct Args {
    int n, n_pad, R;
    int sweep_begin, sweep_end;
    int init;
    uint64_t seed_base;
    const int64_t* row;
    const int32_t* col;
    const void* w;          // CSR weights, or dense n x n_pad matrix
    const int32_t* f0;      // fields of the all-zero assignment
    const double* temp;
    const float* inv_t;
    void* f_state;          // R x n_pad fields
    int8_t* s_state;        // R x n_pad spins
    unsigned long long* ctr;// R draw counters
    long long* energy;      // R energies
};

template <typename F, typename WT, bool DENSE, bool SHARED>
__global__ void __launch_bounds__(32 * kWarpsPerBlock) anneal_kernel(Args a) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int lane = int(threadIdx.x & 31u);
    const int wib = int(threadIdx.x >> 5);
    const int r = int(blockIdx.x) * kWarpsPerBlock + wib;
    if (r >= a.R) return;
    const int n = a.n, np = a.n_pad;
    F* gf = static_cast<F*>(a.f_state) + size_t(r) * size_t(np);
    int8_t* gs = a.s_state + size_t(r) * size_t(np);
    F* f;
    int8_t* s;
    if (SHARED) {
        unsigned char* base = smem + size_t(wib) * size_t(np) * (sizeof(F) + 1);
        f = reinterpret_cast<F*>(base);
        s = reinterpret_cast<int8_t*>(base + size_t(np) * sizeof(F));
    } else {
        f = gf;
        s = gs;
    }
    uint64_t c;
    long long E;
    if (a.init) {
        for (int i = lane; i < n; i += 32) { f[i] = F(a.f0[i]); s[i] = -1; }
        c = 0;
        E = 0;
    } else {
        if (SHARED)
            for (int i = lane; i < n; i += 32) { f[i] = gf[i]; s[i] = gs[i]; }
        c = a.ctr[r];
        E = a.energy[r];
    }
    __syncwarp();
    const uint64_t seed = a.seed_base + uint64_t(r);
    const uint64_t s0 = seed * 0x2545F4914F6CDD1Dull + 1;
    const unsigned below = (1u << lane) - 1u;
    const WT* wv = static_cast<const WT*>(a.w);
    for (int sweep = a.sweep_begin; sweep < a.sweep_end; ++sweep) {
        const double T = a.temp[sweep];
        const float it = a.inv_t[sweep];
        int i0 = 0;
        while (i0 < n) {
            const int i = i0 + lane;
            const bool valid = i < n;
            int dE = 0;
            if (valid) dE = -int(s[i]) * int(f[i]);
            const bool need = valid && dE > 0;
            const unsigned md = __ballot_sync(0xffffffffu, need);
            bool acc = valid && dE <= 0;
            if (need) {
                const uint64_t k = c + uint64_t(__popc(md & below)) + 1;
                const uint64_t u53 = mix64(s0 + k * 0x9E3779B97F4A7C15ull) >> 11;
                acc = metropolis(u53, dE, T, it);
            }
            const unsigned ma = __ballot_sync(0xffffffffu, acc);
            if (ma == 0) {
                c += uint64_t(__popc(md));
                i0 += 32;
                continue;
            }
            const int L = __ffs(int(ma)) - 1;
            c += uint64_t(__popc(md & (L == 31 ? 0xffffffffu : ((2u << L) - 1u))));
            const int j = i0 + L;
            const int dEj = __shfl_sync(0xffffffffu, dE, L);
            const int two_s = 2 * int(s[j]);
            if (DENSE) {
                const WT* rowp = wv + size_t(j) * size_t(np);
                for (int q = lane; q < n; q += 32) f[q] = F(int(f[q]) - two_s * int(rowp[q]));
            } else {
                const int64_t e = a.row[j + 1];
                for (int64_t k = a.row[j] + lane; k < e; k += 32) {
                    const int q = a.col[k];
                    f[q] = F(int(f[q]) - two_s * int(wv[k]));
                }
            }
            __syncwarp();
            if (lane == 0) s[j] = int8_t(-(two_s / 2));
            E += dEj;
            __syncwarp();
            i0 = j + 1;
        }
    }
    if (SHARED)
        for (int i = lane; i < n; i += 32) { gf[i] = f[i]; gs[i] = s[i]; }
    if (lane == 0) {
        a.ctr[r] = c;
        a.energy[r] = E;
    }
}

// Independent energy: cut over the edge list, one block per state.
__global__ void oracle_kernel(const int32_t* eu, const int32_t* ev, const int32_t* ew, int64_t m,
                              const uint8_t* X, int n, long long* energy_out) {
    const uint8_t* x = X + size_t(blockIdx.x) * size_t(n);
    long long acc = 0;
    for (int64_t e = threadIdx.x; e < m; e += blockDim.x)
        if (x[eu[e]] != x[ev[e]]) acc += ew[e];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ long long part[32];
    if ((threadIdx.x & 31u) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        long long t = 0;
        for (unsigned w = 0; w < (blockDim.x + 31) / 32; ++w) t += part[w];
        energy_out[blockIdx.x] = -t;
    }
}

}  // namespace dev

// ---- host side ------------------------------------------------------------------

struct DeviceGraph {
    Plan plan;
    int64_t* row = nullptr;
    int32_t* col = nullptr;
    void* w = nullptr;
    int32_t* f0 = nullptr;
    int32_t *eu = nullptr, *ev = nullptr, *ew = nullptr;

    DeviceGraph(const WeightedGraph& g, const Plan& p) : plan(p) {
        const int n = g.n, np = p.n_pad;
        std::vector<int32_t> f0h(size_t(n), 0);
        for (int i = 0; i < n; ++i) {
            int64_t acc = 0;
            for (int64_t k = g.row[size_t(i)]; k < g.row[size_t(i) + 1]; ++k) acc -= g.w[size_t(k)];
            f0h[size_t(i)] = int32_t(acc);
        }
        GISING_CUDA(cudaMalloc(&f0, sizeof(int32_t) * size_t(n)));
        GISING_CUDA(cudaMemcpy(f0, f0h.data(), sizeof(int32_t) * size_t(n), cudaMemcpyHostToDevice));
        GISING_CUDA(cudaMalloc(&row, sizeof(int64_t) * (size_t(n) + 1)));
        GISING_CUDA(cudaMemcpy(row, g.row.data(), sizeof(int64_t) * (size_t(n) + 1), cudaMemcpyHostToDevice));
        if (p.dense) {
            const size_t cells = size_t(n) * size_t(np);
            if (p.weight_bytes == 1) {
                std::vector<int8_t> d(cells, 0);
                for (int i = 0; i < n; ++i)
                    for (int64_t k = g.row[size_t(i)]; k < g.row[size_t(i) + 1]; ++k)
                        d[size_t(i) * size_t(np) + size_t(g.col[size_t(k)])] = int8_t(g.w[size_t(k)]);
                GISING_CUDA(cudaMalloc(&w, cells));
                GISING_CUDA(cudaMemcpy(w, d.data(), cells, cudaMemcpyHostToDevice));
            } else {
                std::vector<int32_t> d(cells, 0);
                for (int i = 0; i < n; ++i)
                    for (int64_t k = g.row[size_t(i)]; k < g.row[size_t(i) + 1]; ++k)
                        d[size_t(i) * size_t(np) + size_t(g.col[size_t(k)])] = g.w[size_t(k)];
                GISING_CUDA(cudaMalloc(&w, cells * 4));
                GISING_CUDA(cudaMemcpy(w, d.data(), cells * 4, cudaMemcpyHostToDevice));
            }
        } else {
            const size_t nnz = g.col.size();
            GISING_CUDA(cudaMalloc(&col, sizeof(int32_t) * nnz));
            GISING_CUDA(cudaMemcpy(col, g.col.data(), sizeof(int32_t) * nnz, cudaMemcpyHostToDevice));
            if (p.weight_bytes == 1) {
                std::vector<int8_t> d(nnz);
                for (size_t k = 0; k < nnz; ++k) d[k] = int8_t(g.w[k]);
                GISING_CUDA(cudaMalloc(&w, nnz));
                GISING_CUDA(cudaMemcpy(w, d.data(), nnz, cudaMemcpyHostToDevice));
            } else {
                GISING_CUDA(cudaMalloc(&w, nnz * 4));
                GISING_CUDA(cudaMemcpy(w, g.w.data(), nnz * 4, cudaMemcpyHostToDevice));
            }
        }
        const size_t m = size_t(g.m);
        GISING_CUDA(cudaMalloc(&eu, 4 * m));
        GISING_CUDA(cudaMalloc(&ev, 4 * m));
        GISING_CUDA(cudaMalloc(&ew, 4 * m));
        GISING_CUDA(cudaMemcpy(eu, g.eu.data(), 4 * m, cudaMemcpyHostToDevice));
        GISING_CUDA(cudaMemcpy(ev, g.ev.data(), 4 * m, cudaMemcpyHostToDevice));
        GISING_CUDA(cudaMemcpy(ew, g.ew.data(), 4 * m, cudaMemcpyHostToDevice));
    }
    ~DeviceGraph() {
        cudaFree(row); cudaFree(col); cudaFree(w); cudaFree(f0);
        cudaFree(eu); cudaFree(ev); cudaFree(ew);
    }
    DeviceGraph(const DeviceGraph&) = delete;
    DeviceGraph& operator=(const DeviceGraph&) = delete;
};

struct BatchOut {
    int R = 0, n = 0, n_pad = 0;
    std::vector<int8_t> s;          // R x n_pad spins (+-1)
    std::vector<long long> energy;  // R tracked energies
    double wall_s = 0.0;            // ADR-016 W_gpu
    double kernel_ms = 0.0;         // sum of launch times (events)
    int launches = 0;
    int sweeps_per_launch = 0;
    std::vector<uint8_t> x(int r) const {
        std::vector<uint8_t> out(static_cast<size_t>(n));
        const int8_t* p = s.data() + size_t(r) * size_t(n_pad);
        for (int i = 0; i < n; ++i) out[size_t(i)] = p[i] > 0 ? 1u : 0u;
        return out;
    }
};

namespace detail {

template <typename F, typename WT, bool DENSE, bool SHARED>
inline void launch(const dev::Args& a, cudaStream_t st) {
    const size_t smem = SHARED ? size_t(kWarpsPerBlock) * size_t(a.n_pad) * (sizeof(F) + 1) : 0;
    auto k = dev::anneal_kernel<F, WT, DENSE, SHARED>;
    if (smem > 48 * 1024) GISING_CUDA(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
    const int blocks = (a.R + kWarpsPerBlock - 1) / kWarpsPerBlock;
    k<<<blocks, 32 * kWarpsPerBlock, smem, st>>>(a);
}

template <typename F, typename WT>
inline void launch_fw(const Plan& p, const dev::Args& a, cudaStream_t st) {
    if (p.dense) {
        if (p.shared_state) launch<F, WT, true, true>(a, st); else launch<F, WT, true, false>(a, st);
    } else {
        if (p.shared_state) launch<F, WT, false, true>(a, st); else launch<F, WT, false, false>(a, st);
    }
}

template <typename F>
inline void launch_f(const Plan& p, const dev::Args& a, cudaStream_t st) {
    if (p.weight_bytes == 1) launch_fw<F, int8_t>(p, a, st); else launch_fw<F, int32_t>(p, a, st);
}

inline void dispatch(const Plan& p, const dev::Args& a, cudaStream_t st) {
    if (p.field_bytes == 1) launch_f<int8_t>(p, a, st);
    else if (p.field_bytes == 2) launch_f<int16_t>(p, a, st);
    else launch_f<int32_t>(p, a, st);
}

}  // namespace detail

/// Anneals replicas with seeds seed_base .. seed_base+R-1 under Budget b.
/// updates_per_launch bounds the work of one launch (watchdog); it does not change results.
inline BatchOut run_batch(const DeviceGraph& dg, const Budget& b, uint64_t seed_base, int R,
                          double updates_per_launch = 1.0e9) {
    const Plan& p = dg.plan;
    BatchOut out;
    out.R = R;
    out.n = p.n;
    out.n_pad = p.n_pad;
    const int total = b.anneal_sweeps + b.hold_sweeps;
    const std::vector<double> temp = temperatures(b);
    std::vector<float> inv(temp.size());
    for (size_t k = 0; k < temp.size(); ++k) inv[k] = float(1.0 / temp[k]);
    double *d_temp = nullptr;
    float* d_inv = nullptr;
    void* d_f = nullptr;
    int8_t* d_s = nullptr;
    unsigned long long* d_c = nullptr;
    long long* d_e = nullptr;
    GISING_CUDA(cudaMalloc(&d_temp, sizeof(double) * size_t(total)));
    GISING_CUDA(cudaMalloc(&d_inv, sizeof(float) * size_t(total)));
    GISING_CUDA(cudaMemcpy(d_temp, temp.data(), sizeof(double) * size_t(total), cudaMemcpyHostToDevice));
    GISING_CUDA(cudaMemcpy(d_inv, inv.data(), sizeof(float) * size_t(total), cudaMemcpyHostToDevice));
    const size_t cells = size_t(R) * size_t(p.n_pad);
    GISING_CUDA(cudaMalloc(&d_f, cells * size_t(p.field_bytes)));
    GISING_CUDA(cudaMalloc(&d_s, cells));
    GISING_CUDA(cudaMalloc(&d_c, sizeof(unsigned long long) * size_t(R)));
    GISING_CUDA(cudaMalloc(&d_e, sizeof(long long) * size_t(R)));
    out.s.resize(cells);
    out.energy.resize(size_t(R));
    int chunk = int(std::floor(updates_per_launch / (double(R) * double(p.n))));
    chunk = std::max(1, std::min(total, chunk));
    out.sweeps_per_launch = chunk;
    cudaEvent_t e0, e1;
    GISING_CUDA(cudaEventCreate(&e0));
    GISING_CUDA(cudaEventCreate(&e1));
    GISING_CUDA(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    GISING_CUDA(cudaEventRecord(e0));
    dev::Args a{};
    a.n = p.n; a.n_pad = p.n_pad; a.R = R; a.seed_base = seed_base;
    a.row = dg.row; a.col = dg.col; a.w = dg.w; a.f0 = dg.f0;
    a.temp = d_temp; a.inv_t = d_inv;
    a.f_state = d_f; a.s_state = d_s; a.ctr = d_c; a.energy = d_e;
    for (int sb = 0; sb < total; sb += chunk) {
        a.sweep_begin = sb;
        a.sweep_end = std::min(total, sb + chunk);
        a.init = sb == 0 ? 1 : 0;
        detail::dispatch(p, a, 0);
        GISING_CUDA(cudaGetLastError());
        ++out.launches;
    }
    GISING_CUDA(cudaEventRecord(e1));
    GISING_CUDA(cudaMemcpy(out.s.data(), d_s, cells, cudaMemcpyDeviceToHost));
    GISING_CUDA(cudaMemcpy(out.energy.data(), d_e, sizeof(long long) * size_t(R), cudaMemcpyDeviceToHost));
    GISING_CUDA(cudaDeviceSynchronize());
    out.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    float ms = 0.0f;
    GISING_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    out.kernel_ms = ms;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(d_temp); cudaFree(d_inv); cudaFree(d_f); cudaFree(d_s); cudaFree(d_c); cudaFree(d_e);
    return out;
}

/// Device energies of K states (K x n bytes, 0/1), by an independent edge-list kernel.
inline std::vector<long long> device_energies(const DeviceGraph& dg, const std::vector<uint8_t>& X, int K) {
    const int n = dg.plan.n;
    uint8_t* d_x = nullptr;
    long long* d_e = nullptr;
    GISING_CUDA(cudaMalloc(&d_x, size_t(K) * size_t(n)));
    GISING_CUDA(cudaMalloc(&d_e, sizeof(long long) * size_t(K)));
    GISING_CUDA(cudaMemcpy(d_x, X.data(), size_t(K) * size_t(n), cudaMemcpyHostToDevice));
    dev::oracle_kernel<<<K, 256>>>(dg.eu, dg.ev, dg.ew, dg.plan.m, d_x, n, d_e);
    GISING_CUDA(cudaGetLastError());
    std::vector<long long> e(static_cast<size_t>(K));
    GISING_CUDA(cudaMemcpy(e.data(), d_e, sizeof(long long) * size_t(K), cudaMemcpyDeviceToHost));
    cudaFree(d_x);
    cudaFree(d_e);
    return e;
}

}  // namespace gising
}  // namespace argos
