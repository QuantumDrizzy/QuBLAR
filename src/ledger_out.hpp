#pragma once
// A check's numbers, for the Unibit chain (ADR-021).
//
// Entries are recorded as the check runs and written to out/ledger/<check>.json
// when it ends. Doubles are printed with %.17g, so the chain receives the exact
// binary value the check computed. tools/ledger_batch.py turns the fragments of
// one commit into a batch; unibit-chain signs it.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

struct LedgerOut {
    std::string check;
    std::vector<std::string> rows;

    explicit LedgerOut(const char* check_name) : check(check_name) {}

    static std::string esc(const std::string& s) {
        std::string o;
        for (char c : s) {
            if (c == '"' || c == '\\') o += '\\';
            if (c == '\n') { o += "\\n"; continue; }
            o += c;
        }
        return o;
    }

    static std::string dbl(double v) {
        char b[40];
        std::snprintf(b, sizeof(b), "%.17g", v);
        return b;
    }

    /// `inputs` is a JSON object literal, e.g. `{"exposure_log2":25}`.
    void num(const std::string& key, double v, const char* unit, const char* verdict,
             const std::string& source, const std::string& inputs = "{}") {
        if (!std::isfinite(v)) return;  // the chain refuses non-finite values; so do we
        add(key, dbl(v), unit, verdict, source, inputs, "");
    }

    void flag(const std::string& key, bool v, const char* verdict, const std::string& source,
              const std::string& inputs = "{}") {
        add(key, v ? "true" : "false", "1", verdict, source, inputs, "");
    }

    /// A file the batch tool hashes; `value` is what it holds (e.g. a bit count).
    void artifact(const std::string& key, double value, const char* unit, const std::string& path,
                  const std::string& source, const std::string& inputs = "{}") {
        add(key, dbl(value), unit, "PASS", source, inputs, path);
    }

    void write() const {
        std::filesystem::create_directories("out/ledger");
        std::ofstream f("out/ledger/" + check + ".json");
        f << "{\"check\":\"" << check << "\",\"entries\":[\n";
        for (size_t i = 0; i < rows.size(); ++i) f << rows[i] << (i + 1 < rows.size() ? ",\n" : "\n");
        f << "]}\n";
    }

private:
    void add(const std::string& key, const std::string& value, const char* unit, const char* verdict,
             const std::string& source, const std::string& inputs, const std::string& artifact) {
        std::string r = "{\"id\":\"qublar/" + check + "/" + key + "\",\"value\":" + value +
                        ",\"unit\":\"" + unit + "\",\"class\":\"MEASURED\",\"verdict\":\"" + verdict +
                        "\",\"source\":\"" + esc(source) + "\",\"test\":\"src/" + check +
                        ".cu\",\"inputs\":" + inputs;
        if (!artifact.empty()) r += ",\"artifact_path\":\"" + esc(artifact) + "\"";
        rows.push_back(r + "}");
    }
};

/// A bit field as one byte per voxel: 255 outside the domain, round(254 p) inside.
inline void write_cloud_u8(const std::string& path, size_t n_voxels, const std::vector<int>& var_voxel,
                           const std::vector<float>& p) {
    std::vector<uint8_t> g(n_voxels, 255);
    for (size_t i = 0; i < var_voxel.size(); ++i)
        g[var_voxel[i]] = uint8_t(std::lround(254.0 * std::fmin(1.0, std::fmax(0.0, double(p[i])))));
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(g.data()), std::streamsize(g.size()));
}
