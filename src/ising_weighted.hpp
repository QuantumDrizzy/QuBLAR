// =============================================================================
// QuBLAR -- the integer path: per-edge weights, exact energy (ADR-015)
// =============================================================================
// A weighted sparse Ising/MaxCut instance with exact signed-integer weights
// (int32 storage, int64 accumulation) and the annealer of ising_recon.hpp,
// re-expressed on a maintained local field so that each proposal is O(1) and
// every energy is an integer:
//
//   x in {0,1}^n, s = 2x - 1
//   cut(x) = sum_edges w_uv [x_u != x_v]        E(x) = -cut(x)
//   f_i    = sum_j w_ij s_j                     dE_i = -s_i f_i  (flip of i)
//
// The schedule, sweep order and RNG stream are those of argos::anneal_branch
// (a uniform is drawn only when dE > 0), so on a weight +1 graph the accept
// decisions -- and therefore the assignments -- are the same (ADR-015 R1).
// The float path stays argos::BinaryProblem; nothing here touches it.
// Pure host C++17: no CUDA, no dependency beyond the standard library.
// =============================================================================

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace argos {
namespace wising {

struct WeightedGraph {
    int n = 0;
    int64_t m = 0;
    std::vector<int32_t> eu, ev, ew;        // edge list, 0-indexed, as read
    std::vector<int64_t> row;               // CSR offsets, n + 1
    std::vector<int32_t> col, w;            // CSR, both directions
    int64_t sum_w = 0, n_pos = 0, n_neg = 0, n_other = 0;
};

// ---- loader: rudy / Gset text ------------------------------------------------
// Header "n m", then m lines "u v [w]", 1-indexed, missing w = +1. Refuses a
// non-integer weight (that is the float path), a self-loop, a duplicate edge, an
// index out of range, or an edge count that differs from the header.

namespace detail {
struct Cursor {
    const char* p;
    const char* end;
    void skip_blank() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\r')) ++p; }
    void skip_ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p; }
    bool at_eol() { skip_blank(); return p >= end || *p == '\n'; }
    // Reads a token; returns false at end of input.
    bool token(std::string& t) {
        skip_blank();
        if (p >= end || *p == '\n') return false;
        const char* s = p;
        while (p < end && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') ++p;
        t.assign(s, p);
        return true;
    }
};
inline bool parse_int(const std::string& t, int64_t& v) {
    if (t.empty()) return false;
    size_t i = 0;
    bool neg = false;
    if (t[0] == '-' || t[0] == '+') { neg = t[0] == '-'; i = 1; }
    if (i >= t.size()) return false;
    int64_t acc = 0;
    for (; i < t.size(); ++i) {
        if (t[i] < '0' || t[i] > '9') return false;
        acc = acc * 10 + (t[i] - '0');
        if (acc > (int64_t(1) << 40)) return false;
    }
    v = neg ? -acc : acc;
    return true;
}
}  // namespace detail

inline bool parse_rudy(const char* data, size_t size, WeightedGraph& g, std::string& err) {
    g = WeightedGraph();
    detail::Cursor c{data, data + size};
    std::string t;
    int64_t n = 0, m = 0;
    c.skip_ws();
    if (!c.token(t) || !detail::parse_int(t, n) || n <= 0 || n > (int64_t(1) << 30)) {
        err = "bad header: n"; return false;
    }
    if (!c.token(t) || !detail::parse_int(t, m) || m < 0) { err = "bad header: m"; return false; }
    g.n = int(n);
    g.m = m;
    g.eu.reserve(size_t(m)); g.ev.reserve(size_t(m)); g.ew.reserve(size_t(m));
    int64_t read = 0;
    while (true) {
        c.skip_ws();
        if (c.p >= c.end) break;
        int64_t u = 0, v = 0, w = 1;
        if (!c.token(t) || !detail::parse_int(t, u)) { err = "bad vertex token '" + t + "'"; return false; }
        if (!c.token(t) || !detail::parse_int(t, v)) { err = "bad vertex token '" + t + "'"; return false; }
        if (c.token(t)) {
            if (!detail::parse_int(t, w)) {
                err = "weight '" + t + "' is not an integer; the float path is BinaryProblem";
                return false;
            }
            if (w < std::numeric_limits<int32_t>::min() || w > std::numeric_limits<int32_t>::max()) {
                err = "weight out of int32 range"; return false;
            }
            if (!c.at_eol()) { err = "trailing token on an edge line"; return false; }
        }
        if (u < 1 || u > n || v < 1 || v > n) { err = "vertex index out of range"; return false; }
        if (u == v) { err = "self-loop"; return false; }
        g.eu.push_back(int32_t(u - 1));
        g.ev.push_back(int32_t(v - 1));
        g.ew.push_back(int32_t(w));
        ++read;
    }
    if (read != m) {
        err = "header says " + std::to_string(m) + " edges, file has " + std::to_string(read);
        return false;
    }
    // duplicates, in either orientation
    std::vector<uint64_t> key(static_cast<size_t>(m));
    for (int64_t e = 0; e < m; ++e) {
        const uint64_t a = uint64_t(std::min(g.eu[e], g.ev[e]));
        const uint64_t b = uint64_t(std::max(g.eu[e], g.ev[e]));
        key[size_t(e)] = (a << 32) | b;
    }
    std::sort(key.begin(), key.end());
    for (size_t k = 1; k < key.size(); ++k)
        if (key[k] == key[k - 1]) { err = "duplicate edge"; return false; }
    // CSR, both directions
    g.row.assign(size_t(n) + 1, 0);
    for (int64_t e = 0; e < m; ++e) { g.row[size_t(g.eu[e]) + 1]++; g.row[size_t(g.ev[e]) + 1]++; }
    for (int64_t i = 0; i < n; ++i) g.row[size_t(i) + 1] += g.row[size_t(i)];
    g.col.resize(size_t(2 * m));
    g.w.resize(size_t(2 * m));
    std::vector<int64_t> fill(g.row.begin(), g.row.end() - 1);
    for (int64_t e = 0; e < m; ++e) {
        const int32_t a = g.eu[e], b = g.ev[e], w = g.ew[e];
        g.col[size_t(fill[a])] = b; g.w[size_t(fill[a]++)] = w;
        g.col[size_t(fill[b])] = a; g.w[size_t(fill[b]++)] = w;
        g.sum_w += w;
        if (w == 1) ++g.n_pos; else if (w == -1) ++g.n_neg; else ++g.n_other;
    }
    return true;
}

inline bool read_file(const std::string& path, std::vector<char>& buf) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
#if defined(_WIN32)
    _fseeki64(f, 0, SEEK_END);
    const long long size = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
#else
    fseeko(f, 0, SEEK_END);
    const long long size = static_cast<long long>(ftello(f));
    fseeko(f, 0, SEEK_SET);
#endif
    if (size < 0) { std::fclose(f); return false; }
    buf.resize(size_t(size));
    const size_t got = size > 0 ? std::fread(buf.data(), 1, size_t(size), f) : 0;
    std::fclose(f);
    return got == size_t(size);
}

// ---- exact energies ------------------------------------------------------------

/// cut from the edge list as read: the independent count.
inline int64_t cut_of(const WeightedGraph& g, const std::vector<uint8_t>& x) {
    int64_t cut = 0;
    for (int64_t e = 0; e < g.m; ++e)
        if (x[size_t(g.eu[e])] != x[size_t(g.ev[e])]) cut += g.ew[e];
    return cut;
}

/// E = -cut, from scratch over the CSR (each edge seen once, j > i).
inline int64_t energy_scratch(const WeightedGraph& g, const std::vector<uint8_t>& x) {
    int64_t cut = 0;
    for (int i = 0; i < g.n; ++i)
        for (int64_t k = g.row[size_t(i)]; k < g.row[size_t(i) + 1]; ++k) {
            const int j = g.col[size_t(k)];
            if (j > i && x[size_t(i)] != x[size_t(j)]) cut += g.w[size_t(k)];
        }
    return -cut;
}

// ---- state with a maintained local field ------------------------------------

struct State {
    const WeightedGraph* g = nullptr;
    std::vector<int8_t> s;      // +-1
    std::vector<int64_t> f;     // f_i = sum_j w_ij s_j
    int64_t energy = 0;         // E = -cut, tracked

    explicit State(const WeightedGraph& graph) : g(&graph), s(size_t(graph.n), int8_t(-1)),
                                                 f(size_t(graph.n), 0) {
        // x = 0 everywhere: s = -1, cut = 0
        for (int i = 0; i < graph.n; ++i) {
            int64_t acc = 0;
            for (int64_t k = graph.row[size_t(i)]; k < graph.row[size_t(i) + 1]; ++k)
                acc -= graph.w[size_t(k)];
            f[size_t(i)] = acc;
        }
    }
    int64_t dE(int i) const { return -int64_t(s[size_t(i)]) * f[size_t(i)]; }
    void flip(int i) {
        const int64_t d = dE(i);
        const int64_t si = s[size_t(i)];
        for (int64_t k = g->row[size_t(i)]; k < g->row[size_t(i) + 1]; ++k)
            f[size_t(g->col[size_t(k)])] -= 2 * int64_t(g->w[size_t(k)]) * si;
        s[size_t(i)] = int8_t(-si);
        energy += d;
    }
    std::vector<uint8_t> x() const {
        std::vector<uint8_t> out(s.size());
        for (size_t i = 0; i < s.size(); ++i) out[i] = s[i] > 0 ? 1u : 0u;
        return out;
    }
};

// ---- the annealer: anneal_branch's schedule and RNG, integer dE ---------------

struct Budget {
    double t_hot = 5.0;
    double t_cold = 1e-3;
    int anneal_sweeps = 400;
    int hold_sweeps = 20;
};

namespace detail {
// Identical to argos::detail::splitmix / uniform in ising_recon.hpp (checked by R1).
inline uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
inline double uniform(uint64_t& s) { return (splitmix(s) >> 11) * 0x1.0p-53; }
}  // namespace detail

struct RunOut {
    std::vector<uint8_t> x;
    int64_t energy_tracked = 0;
};

inline RunOut anneal(const WeightedGraph& g, const Budget& b, uint64_t seed) {
    State st(g);
    uint64_t state = seed * 0x2545F4914F6CDD1Dull + 1;
    const int total = b.anneal_sweeps + b.hold_sweeps;
    for (int sweep = 0; sweep < total; ++sweep) {
        const double frac = std::min(1.0, double(sweep) / std::max(1, b.anneal_sweeps - 1));
        const double temp = sweep < b.anneal_sweeps
            ? b.t_hot * std::pow(b.t_cold / b.t_hot, frac) : b.t_cold;
        for (int i = 0; i < g.n; ++i) {
            const double dE = double(st.dE(i));
            if (dE <= 0.0 || detail::uniform(state) < std::exp(-dE / temp)) st.flip(i);
        }
    }
    RunOut out;
    out.x = st.x();
    out.energy_tracked = st.energy;
    return out;
}

// ---- TTS99 and its bootstrap CI (ADR-015 section 5) ---------------------------

inline double tts99(double t_mean, double p) {
    if (p <= 0.0) return std::numeric_limits<double>::infinity();
    if (p >= 0.99) return t_mean;
    return t_mean * std::log(0.01) / std::log(1.0 - p);
}

struct TtsCi { double lo, hi; };

inline TtsCi tts99_bootstrap(const std::vector<int>& success, const std::vector<double>& time,
                             int resamples = 2000, uint64_t seed = 20260928ull) {
    const size_t r = success.size();
    std::vector<double> v;
    v.reserve(size_t(resamples));
    uint64_t st = seed;
    for (int b = 0; b < resamples; ++b) {
        int s = 0;
        double t = 0.0;
        for (size_t k = 0; k < r; ++k) {
            const size_t j = size_t(detail::splitmix(st) % uint64_t(r));
            s += success[j];
            t += time[j];
        }
        v.push_back(tts99(t / double(r), double(s) / double(r)));
    }
    std::sort(v.begin(), v.end());   // inf sorts last
    auto pct = [&](double q) {
        const double pos = q * double(v.size() - 1);
        const size_t lo = size_t(std::floor(pos)), hi = size_t(std::ceil(pos));
        if (std::isinf(v[lo]) || std::isinf(v[hi])) return std::numeric_limits<double>::infinity();
        return v[lo] + (v[hi] - v[lo]) * (pos - double(lo));
    };
    return {pct(0.025), pct(0.975)};
}

// ---- SHA-256 (for assignment and file hashes) ----------------------------------

class Sha256 {
public:
    Sha256() { reset(); }
    void reset() {
        static const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::memcpy(h_, init, sizeof h_);
        len_ = 0; fill_ = 0;
    }
    void update(const void* data, size_t n) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        len_ += uint64_t(n);
        while (n > 0) {
            const size_t take = std::min(n, size_t(64) - fill_);
            std::memcpy(buf_ + fill_, p, take);
            fill_ += take; p += take; n -= take;
            if (fill_ == 64) { block(buf_); fill_ = 0; }
        }
    }
    std::string hex() {
        uint8_t pad[128] = {0x80};
        const uint64_t bits = len_ * 8;
        const size_t padlen = (fill_ < 56) ? 56 - fill_ : 120 - fill_;
        update(pad, padlen);
        uint8_t lenb[8];
        for (int i = 0; i < 8; ++i) lenb[i] = uint8_t(bits >> (56 - 8 * i));
        update(lenb, 8);
        static const char* d = "0123456789abcdef";
        std::string out;
        for (uint32_t v : h_)
            for (int k = 28; k >= 0; k -= 4) out.push_back(d[(v >> k) & 15]);
        return out;
    }
    static std::string of(const void* data, size_t n) { Sha256 s; s.update(data, n); return s.hex(); }

private:
    static uint32_t rotr(uint32_t x, int k) { return (x >> k) | (x << (32 - k)); }
    void block(const uint8_t* b) {
        static const uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = uint32_t(b[4 * i]) << 24 | uint32_t(b[4 * i + 1]) << 16 |
                   uint32_t(b[4 * i + 2]) << 8 | uint32_t(b[4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], bb = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & bb) ^ (a & c) ^ (bb & c);
            const uint32_t t2 = S0 + mj;
            h = g; g = f; f = e; e = d + t1; d = c; c = bb; bb = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += bb; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
    }
    uint32_t h_[8];
    uint8_t buf_[64];
    uint64_t len_;
    size_t fill_;
};

inline std::string assignment_sha256(const std::vector<uint8_t>& x) {
    std::string bits(x.size(), '0');
    for (size_t i = 0; i < x.size(); ++i) bits[i] = x[i] ? '1' : '0';
    return Sha256::of(bits.data(), bits.size());
}

}  // namespace wising
}  // namespace argos
