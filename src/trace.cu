// =============================================================================
// QuBLAR -- correctness harness for the CUDA BVH traversal
// =============================================================================
// Rays in, nearest hits out. This is Phase 1 of ADR-001 and it is the baseline the
// OptiX path will be measured against, so the two must agree hit for hit before any
// speedup number is allowed to mean something.
//
// Correctness is checked three ways, because a tracer that is subtly wrong produces
// plausible garbage forever:
//
//   1. analytic  -- rays at a plane of known range and tilt, where the answer is
//                   arithmetic and not an opinion
//   2. brute force -- every ray against every triangle, no hierarchy at all. If the
//                   BVH disagrees with this, the BVH is wrong. It is O(N*M) and only
//                   runs on small scenes, which is the point of small scenes.
//   3. watertightness -- rays fired exactly along shared triangle edges, where a naive
//                   intersector leaks and reports a miss through solid geometry
// =============================================================================

#include "trace.cuh"

static std::vector<Hit> run(const DeviceScene& ds, const std::vector<Ray>& rays, bool brute) {
    Ray* d_rays = nullptr; Hit* d_hits = nullptr;
    const int n = static_cast<int>(rays.size());
    CUDA_CHECK(cudaMalloc(&d_rays, n * sizeof(Ray)));
    CUDA_CHECK(cudaMalloc(&d_hits, n * sizeof(Hit)));
    CUDA_CHECK(cudaMemcpy(d_rays, rays.data(), n * sizeof(Ray), cudaMemcpyHostToDevice));

    const int block = 128, grid = (n + block - 1) / block;
    if (brute) trace_brute<<<grid, block>>>(ds.tris, ds.n_tris, d_rays, d_hits, n);
    else       trace_bvh<<<grid, block>>>(ds.nodes, ds.indices, ds.tris, d_rays, d_hits, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Hit> hits(n);
    CUDA_CHECK(cudaMemcpy(hits.data(), d_hits, n * sizeof(Hit), cudaMemcpyDeviceToHost));
    cudaFree(d_rays); cudaFree(d_hits);
    return hits;
}

static int failures = 0;
static void check(bool ok, const char* what, const std::string& detail = "") {
    std::printf("  %-46s %s%s%s\n", what, ok ? "PASS" : "FAIL",
                detail.empty() ? "" : "  ", detail.c_str());
    if (!ok) failures++;
}

int main() {
    std::printf("QuBLAR Phase 1 -- CUDA BVH traversal\n\n");

    // ---- 1. analytic: a tilted plane has a closed-form hit distance ----------
    {
        const float d = 10.0f, tilt = 30.0f;
        Scene s = make_tilted_plane(d, tilt);
        DeviceScene ds; ds.upload(s);

        std::vector<Ray> rays;
        std::vector<float> expected;
        const float ta = std::tan(tilt * 3.14159265358979f / 180.f);
        for (int i = -8; i <= 8; ++i) {
            const float x = i * 0.5f;
            rays.push_back({make_float3(x, 0.f, 0.f), make_float3(0.f, 0.f, 1.f), 1e30f});
            expected.push_back(d + x * ta);   // plane: z = d + x*tan(a)
        }

        const auto hits = run(ds, rays, false);
        float worst = 0.f;
        for (size_t i = 0; i < rays.size(); ++i)
            worst = std::max(worst, std::fabs(hits[i].t - expected[i]));

        check(worst < 1e-3f, "analytic tilted plane, 17 rays",
              "max |error| = " + std::to_string(worst) + " m");

        // cos(theta) between the ray and the surface normal must equal cos(tilt).
        const float expect_cos = std::cos(tilt * 3.14159265358979f / 180.f);
        float worst_cos = 0.f;
        for (const auto& h : hits) {
            const float c = std::fabs(h.normal.z);        // ray is along +z
            worst_cos = std::max(worst_cos, std::fabs(c - expect_cos));
        }
        check(worst_cos < 1e-4f, "surface normal matches the known tilt",
              "max |error| = " + std::to_string(worst_cos));
        ds.free();
    }

    // ---- 2. the BVH must agree with brute force, hit for hit ----------------
    {
        Scene s = make_box(4.0f, 6);
        DeviceScene ds; ds.upload(s);
        std::printf("  [box: %zu triangles, %zu BVH nodes]\n", s.triangles.size(), s.nodes.size());

        // STRUCTURAL CHECK, and it exists because its absence let a real bug through.
        //
        // An empty AABB poisoned the SAH accumulator, so every split costed as infinite
        // and the builder returned ONE node holding all 432 triangles. The agreement test
        // below still passed -- a single-leaf "hierarchy" IS brute force, so the two
        // tracers agreed perfectly while measuring nothing. A test a broken
        // implementation passes is not a test.
        size_t leaves = 0, biggest_leaf = 0, in_leaves = 0;
        for (const auto& n : s.nodes)
            if (n.count > 0) {
                leaves++;
                in_leaves += n.count;
                biggest_leaf = std::max(biggest_leaf, static_cast<size_t>(n.count));
            }
        check(s.nodes.size() > 1, "the BVH actually branches",
              std::to_string(s.nodes.size()) + " nodes, " + std::to_string(leaves) + " leaves");
        check(biggest_leaf <= 4, "no leaf exceeds the maximum",
              "largest holds " + std::to_string(biggest_leaf));
        check(in_leaves == s.triangles.size(), "every triangle referenced exactly once",
              std::to_string(in_leaves) + " of " + std::to_string(s.triangles.size()));

        // Rays from outside in every direction, deterministic so a failure reproduces.
        std::vector<Ray> rays;
        unsigned seed = 12345u;
        auto rnd = [&]() {
            seed = seed * 1664525u + 1013904223u;
            return (seed >> 8) * (1.0f / 16777216.0f);     // [0,1)
        };
        for (int i = 0; i < 20000; ++i) {
            const float th = rnd() * 6.283185f, ph = std::acos(1.f - 2.f * rnd());
            const Vec3 o{8.f * std::sin(ph) * std::cos(th), 8.f * std::sin(ph) * std::sin(th),
                         8.f * std::cos(ph)};
            const Vec3 target{(rnd() - 0.5f) * 3.f, (rnd() - 0.5f) * 3.f, (rnd() - 0.5f) * 3.f};
            const Vec3 dir = normalize(target - o);
            rays.push_back({make_float3(o.x, o.y, o.z), make_float3(dir.x, dir.y, dir.z), 1e30f});
        }

        const auto a = run(ds, rays, false);
        const auto b = run(ds, rays, true);

        int mismatched = 0; float worst_t = 0.f;
        for (size_t i = 0; i < rays.size(); ++i) {
            const bool hit_a = a[i].t >= 0.f, hit_b = b[i].t >= 0.f;
            if (hit_a != hit_b) { mismatched++; continue; }
            if (hit_a) worst_t = std::max(worst_t, std::fabs(a[i].t - b[i].t));
        }
        check(mismatched == 0, "BVH agrees with brute force on 20000 rays",
              std::to_string(mismatched) + " disagreements");
        check(worst_t < 1e-4f, "distances identical where both hit",
              "max |delta| = " + std::to_string(worst_t) + " m");
        ds.free();
    }

    // ---- 3. watertightness: no ray may escape a closed box from inside -------
    {
        Scene s = make_box(4.0f, 8);
        DeviceScene ds; ds.upload(s);

        std::vector<Ray> rays;
        unsigned seed = 999u;
        auto rnd = [&]() {
            seed = seed * 1664525u + 1013904223u;
            return (seed >> 8) * (1.0f / 16777216.0f);
        };
        // Fired from the exact centre, including straight down the axes and diagonals,
        // which are where a mesh's shared edges line up with the ray.
        for (int i = 0; i < 30000; ++i) {
            float dx, dy, dz;
            if (i < 6) {                                   // the six axis directions
                dx = (i == 0) - (i == 1); dy = (i == 2) - (i == 3); dz = (i == 4) - (i == 5);
            } else if (i < 14) {                           // the eight corner diagonals
                const int b = i - 6;
                const float k = 0.57735027f;
                dx = (b & 1) ? k : -k; dy = (b & 2) ? k : -k; dz = (b & 4) ? k : -k;
            } else {
                const float th = rnd() * 6.283185f, ph = std::acos(1.f - 2.f * rnd());
                dx = std::sin(ph) * std::cos(th); dy = std::sin(ph) * std::sin(th); dz = std::cos(ph);
            }
            rays.push_back({make_float3(0.f, 0.f, 0.f), make_float3(dx, dy, dz), 1e30f});
        }

        const auto hits = run(ds, rays, false);
        int escaped = 0;
        for (const auto& h : hits) if (h.t < 0.f) escaped++;
        check(escaped == 0, "no ray escapes a closed box (30000 rays)",
              std::to_string(escaped) + " leaks");
        ds.free();
    }

    std::printf("\n%s\n", failures == 0 ? "all checks passed" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
