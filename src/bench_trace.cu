// =============================================================================
// QuBLAR -- CUDA BVH against RT cores, on identical input
// =============================================================================
// ADR-002 items 4 and 5. Agreement first: if the two tracers do not describe the same
// surfaces, whichever is faster is faster at something else. Only then, timing.
//
// What is compared is TRAVERSAL, and nothing else. The sensor model is not ported --
// OptiX programs cannot use shared memory, and the LiDAR kernel accumulates its waveform
// there, so a ported version would be measuring the memory system. The resulting number
// is therefore an upper bound on what the full simulator could gain, not a prediction of
// it. Any claim phrased as "QuBLAR is N times faster" would be false; the true claim
// names the traversal.
//
// Three things this harness refuses to do, each because it is the standard way such a
// number gets inflated:
//
//   - report one ratio. RT cores win more as the hierarchy deepens, so a single scene
//     says as much about the scene as about the hardware. The sweep is the result.
//   - use only coherent rays. Divergent rays are much harder for both sides and the gap
//     between the two regimes is large.
//   - fold the acceleration build into the traversal time. OptiX builds in hardware; the
//     baseline builds on the host. They are different costs and are printed apart.
// =============================================================================

#include <optix.h>
// Defines g_optixFunctionTable, which optix_stubs.h only declares. It must appear in
// exactly one translation unit; putting it in the header would break the moment a
// second .cu included the tracer.
#include <optix_function_table_definition.h>

#include "trace.cuh"
#include "optix_tracer.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace argos;

static int failures = 0;

static void check(bool ok, const char* what, const std::string& detail = "") {
    std::printf("    %-44s %s  %s\n", what, ok ? "PASS" : "FAIL", detail.c_str());
    if (!ok) failures++;
}

// -----------------------------------------------------------------------------
// Rays
// -----------------------------------------------------------------------------

static unsigned wang(unsigned x) {
    x = (x ^ 61u) ^ (x >> 16); x *= 9u; x ^= x >> 4; x *= 0x27d4eb2du; x ^= x >> 15;
    return x;
}
static float unit(unsigned x) { return (wang(x) >> 8) * (1.0f / 16777216.0f); }

/// Rays from the centre of the scene outward. `spread` is the half-angle of the cone
/// they occupy: a narrow cone is what a scanner fires and neighbouring threads follow
/// nearly the same path through the tree; the full sphere is the opposite, and both
/// tracers do much worse on it. Reporting only the first is how a traversal benchmark
/// flatters itself.
static std::vector<Ray> make_rays(int n, float spread_cos, unsigned seed) {
    std::vector<Ray> rays(n);
    for (int i = 0; i < n; ++i) {
        const unsigned k = static_cast<unsigned>(i) ^ seed;
        const float u1 = unit(k * 0x9E3779B9u);
        const float u2 = unit(k * 0x85EBCA6Bu + 7u);
        const float cz = 1.0f - u1 * (1.0f - spread_cos);
        const float sz = std::sqrt(std::max(0.0f, 1.0f - cz * cz));
        const float ph = 6.28318530718f * u2;
        rays[i].origin = make_float3(0.f, 0.f, 0.f);
        rays[i].direction = make_float3(sz * std::cos(ph), sz * std::sin(ph), cz);
        rays[i].tmax = 1.0e16f;
    }
    return rays;
}

// -----------------------------------------------------------------------------
// Timing
// -----------------------------------------------------------------------------

/// Median, not mean. A single scheduler preemption moves a mean by more than the effect
/// being measured and does not move a median at all.
template <typename F>
static double median_ms(F&& launch, int warmup = 3, int reps = 11) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> ms(reps);
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(t0));
        launch();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], t0, t1));
    }
    std::sort(ms.begin(), ms.end());
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    return ms[reps / 2];
}

// -----------------------------------------------------------------------------

struct Agreement {
    int    hit_mismatch = 0;     // one found a surface where the other found none
    int    bvh_only = 0;         // and which one -- a disagreement is not symmetric
    int    rt_only = 0;
    int    tri_differs = 0;      // same surface, different triangle id -- legal at edges
    double max_dt = 0.0;
    double min_ndot = 1.0;
};

static Agreement compare(const std::vector<Hit>& a, const std::vector<Hit>& b) {
    Agreement g;
    for (size_t i = 0; i < a.size(); ++i) {
        const bool ha = a[i].t > 0.f, hb = b[i].t > 0.f;
        if (ha != hb) { g.hit_mismatch++; (ha ? g.bvh_only : g.rt_only)++; continue; }
        if (!ha) continue;
        g.max_dt = std::max(g.max_dt, std::fabs(double(a[i].t) - double(b[i].t)));
        const double nd = double(a[i].normal.x) * b[i].normal.x
                        + double(a[i].normal.y) * b[i].normal.y
                        + double(a[i].normal.z) * b[i].normal.z;
        g.min_ndot = std::min(g.min_ndot, nd);
        if (a[i].triangle != b[i].triangle) g.tri_differs++;
    }
    return g;
}

// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    const std::string module_path = argc > 1 ? argv[1] : "build/optix_programs.optixir";
    const int n_rays = 1 << 20;

    std::printf("\nQuBLAR Phase 2 -- CUDA BVH against RT cores\n\n");

    OptixTracer rt;
    rt.init(module_path);
    std::printf("  RT core version %u, %d rays per launch\n\n", rt.rtcore_version(), n_rays);

    // Two ray regimes, same origin, differing only in angular spread.
    const std::vector<Ray> coherent   = make_rays(n_rays, std::cos(0.05f), 0x1234u);
    const std::vector<Ray> incoherent = make_rays(n_rays, -1.0f, 0x5678u);

    Ray* d_rays = nullptr; Hit* d_hits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rays, n_rays * sizeof(Ray)));
    CUDA_CHECK(cudaMalloc(&d_hits, n_rays * sizeof(Hit)));

    std::printf("  %-9s %-12s %-12s %-12s %-9s %-12s\n",
                "triangles", "rays", "bvh Mray/s", "optix Mray/s", "speedup", "edge ties");
    std::printf("  %s\n", std::string(74, '-').c_str());

    for (int subdiv : {6, 16, 32, 64, 96}) {
        Scene scene = make_box(4.0f, subdiv);
        const int n_tris = static_cast<int>(scene.triangles.size());

        DeviceScene ds; ds.upload(scene);
        OptixTracer* tracer = &rt;
        tracer->build(scene);

        const int block = 128, grid = (n_rays + block - 1) / block;

        for (int regime = 0; regime < 2; ++regime) {
            const std::vector<Ray>& rays = regime == 0 ? coherent : incoherent;
            CUDA_CHECK(cudaMemcpy(d_rays, rays.data(), n_rays * sizeof(Ray),
                                  cudaMemcpyHostToDevice));

            // --- agreement, before either side is timed ------------------------
            std::vector<Hit> h_bvh(n_rays), h_rt(n_rays);
            trace_bvh<<<grid, block>>>(ds.nodes, ds.indices, ds.tris, d_rays, d_hits, n_rays);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_bvh.data(), d_hits, n_rays * sizeof(Hit),
                                  cudaMemcpyDeviceToHost));
            tracer->trace(d_rays, d_hits, n_rays);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_rt.data(), d_hits, n_rays * sizeof(Hit),
                                  cudaMemcpyDeviceToHost));

            const Agreement g = compare(h_bvh, h_rt);
            if (g.hit_mismatch != 0 || g.max_dt > 1e-3 || g.min_ndot < 0.999) {
                std::printf("\n  [%d triangles, %s rays] tracers disagree\n",
                            n_tris, regime == 0 ? "coherent" : "incoherent");
                check(g.hit_mismatch == 0, "hit and miss agree",
                      std::to_string(g.hit_mismatch) + " of " + std::to_string(n_rays)
                      + "  (bvh only " + std::to_string(g.bvh_only)
                      + ", optix only " + std::to_string(g.rt_only) + ")");
                check(g.max_dt <= 1e-3, "distances agree",
                      "max |delta| = " + std::to_string(g.max_dt) + " m");
                check(g.min_ndot >= 0.999, "normals agree",
                      "min n.n = " + std::to_string(g.min_ndot));
            }

            const double ms_bvh = median_ms([&] {
                trace_bvh<<<grid, block>>>(ds.nodes, ds.indices, ds.tris, d_rays, d_hits, n_rays);
            });
            const double ms_rt = median_ms([&] { tracer->trace(d_rays, d_hits, n_rays); });

            std::printf("  %-9d %-12s %-12.1f %-12.1f %-9.2f %-12zu\n",
                        n_tris, regime == 0 ? "coherent" : "incoherent",
                        n_rays / ms_bvh * 1e-3, n_rays / ms_rt * 1e-3,
                        ms_bvh / ms_rt, static_cast<size_t>(g.tri_differs));
        }

        std::printf("  %-9s %-12s host BVH %zu nodes = %zu B, depth %d of %d, optix build %.2f ms\n",
                    "", "[build]", scene.nodes.size(),
                    scene.nodes.size() * sizeof(BvhNode),
                    scene.max_depth, kStackSize, rt.build_ms);

        ds.free();
        tracer->free_scene();
    }

    // --- watertightness through the hardware intersector -----------------------
    // The baseline needed a fix to pass this (a NaN in the slab test). OptiX's triangle
    // intersection is watertight by construction, so a leak HERE would mean the scene is
    // open rather than the intersector -- which would retroactively invalidate the
    // baseline's clean result as well.
    {
        std::printf("\n  watertightness through RT cores\n");
        Scene scene = make_box(4.0f, 8);
        rt.build(scene);

        const int n = 30000;
        std::vector<Ray> rays(n);
        for (int i = 0; i < n; ++i) {
            const float u1 = unit(static_cast<unsigned>(i) * 0x9E3779B9u);
            const float u2 = unit(static_cast<unsigned>(i) * 0x85EBCA6Bu + 11u);
            const float cz = 1.0f - 2.0f * u1;
            const float sz = std::sqrt(std::max(0.0f, 1.0f - cz * cz));
            const float ph = 6.28318530718f * u2;
            rays[i].origin = make_float3(0.f, 0.f, 0.f);
            rays[i].direction = make_float3(sz * std::cos(ph), sz * std::sin(ph), cz);
            rays[i].tmax = 1.0e16f;
        }
        // The six axis rays that exposed the baseline's NaN, fired explicitly.
        const float3 axes[6] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
        for (int i = 0; i < 6; ++i) rays[i].direction = axes[i];

        CUDA_CHECK(cudaMemcpy(d_rays, rays.data(), n * sizeof(Ray), cudaMemcpyHostToDevice));
        rt.trace(d_rays, d_hits, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<Hit> hits(n);
        CUDA_CHECK(cudaMemcpy(hits.data(), d_hits, n * sizeof(Hit), cudaMemcpyDeviceToHost));

        int leaks = 0;
        for (int i = 0; i < n; ++i) if (hits[i].t <= 0.f) leaks++;
        check(leaks == 0, "no ray escapes a closed box",
              std::to_string(leaks) + " leaks of " + std::to_string(n));
        rt.free_scene();
    }

    cudaFree(d_rays); cudaFree(d_hits);
    std::printf("\n%s\n\n", failures == 0 ? "tracers agree" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
