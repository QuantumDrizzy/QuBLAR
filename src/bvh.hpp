// =============================================================================
// QuBLAR -- scene geometry and a bounding volume hierarchy
// =============================================================================
// The BVH here is deliberately ordinary: a binary tree over triangles, built on the
// host by binned surface-area heuristic, flattened into an array the GPU walks.
//
// It is ordinary on purpose. This is the BASELINE (ADR-001): the number the RT-core
// path will be measured against. A clever hand-written traversal would make the
// comparison flattering to the wrong side and would take work away from the physics,
// which is where this project's risk actually lives. What it must be is *correct*,
// because both tracers have to agree hit for hit before any speedup means anything.
// =============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace argos {

struct Vec3 {
    float x = 0.f, y = 0.f, z = 0.f;

    Vec3() = default;
    Vec3(float a, float b, float c) : x(a), y(b), z(c) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
};

inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline Vec3 normalize(const Vec3& v) {
    const float n = std::sqrt(dot(v, v));
    return n > 0.f ? Vec3{v.x / n, v.y / n, v.z / n} : v;
}

/// One triangle, plus the id of the material that governs how it returns light.
struct Triangle {
    Vec3 v0, v1, v2;
    int  material = 0;
};

/// Reflectance at the laser wavelength, which is NOT the visible colour of a surface.
/// Kept as a separate table so a scene can be re-scanned at another wavelength without
/// touching its geometry (ADR-001 leaves the choice of 905 vs 1550 nm open).
struct Material {
    float reflectance = 0.5f;   // rho, [0, 1]
    float roughness   = 0.0f;   // reserved: widens the return in time
};

struct Aabb {
    Vec3 lo{ std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max() };
    Vec3 hi{ -std::numeric_limits<float>::max(),
             -std::numeric_limits<float>::max(),
             -std::numeric_limits<float>::max() };

    void expand(const Vec3& p) {
        lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
        hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
    /// An AABB with no contents has lo = +inf and hi = -inf. Expanding by one of those
    /// without checking sets this box's hi to +inf and its lo to -inf -- the accumulator
    /// is destroyed and every SAH split downstream is costed against an infinite area, so
    /// the leaf always wins and the tree collapses to a single node. The guard is the
    /// whole reason the hierarchy exists.
    bool empty() const { return lo.x > hi.x; }
    void expand(const Aabb& b) { if (b.empty()) return; expand(b.lo); expand(b.hi); }

    Vec3 extent() const { return hi - lo; }

    /// Half the surface area, which is all the SAH needs since the constant cancels.
    float half_area() const {
        const Vec3 e = extent();
        if (e.x < 0.f) return 0.f;             // empty box
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }

    Vec3 centroid() const { return (lo + hi) * 0.5f; }
};

/// Flattened node. A leaf carries a run of triangle indices; an interior node carries
/// its right child's offset, with the left child implicitly the next node.
///
/// 32 bytes, which matters: traversal is pointer-chasing and the node array is what the
/// cache holds. Two AABB floats short of a cache line is not an accident.
struct alignas(16) BvhNode {
    Vec3  lo;
    int   left_first;   // leaf: first triangle index. interior: index of the right child.
    Vec3  hi;
    int   count;        // leaf: triangle count (> 0). interior: 0.
};

static_assert(sizeof(BvhNode) == 32, "BvhNode must stay 32 B for the traversal's sake");

struct Scene {
    std::vector<Triangle> triangles;
    std::vector<Material> materials;

    /// Flattened hierarchy, valid after build().
    std::vector<BvhNode> nodes;
    /// Triangle order the leaves index into -- the build permutes, the geometry does not.
    std::vector<int> indices;

    /// Deepest path from the root, in nodes. The GPU traversal carries a fixed-size
    /// stack, and this is the number that decides whether that stack is big enough.
    /// Kept on the scene rather than recomputed, because a traversal that overflows its
    /// stack does not crash -- it silently returns a miss through solid geometry.
    int max_depth = 0;

    void add_material(float reflectance, float roughness = 0.f) {
        materials.push_back({reflectance, roughness});
    }

    /// A quad as two triangles, wound so the normal points along +n by the right-hand rule.
    void add_quad(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d, int material) {
        triangles.push_back({a, b, c, material});
        triangles.push_back({a, c, d, material});
    }

    void build();
};

// -----------------------------------------------------------------------------
// Build
// -----------------------------------------------------------------------------
namespace detail {

inline Aabb tri_bounds(const Triangle& t) {
    Aabb b;
    b.expand(t.v0);
    b.expand(t.v1);
    b.expand(t.v2);
    return b;
}

/// Binned SAH split.
///
/// The alternative, splitting at the median, is two lines shorter and produces a tree
/// that costs roughly twice as much to traverse on scenes with any size variation --
/// because it balances triangle COUNT while traversal cost is driven by surface AREA.
/// Since this tree is the baseline in a measured comparison, an avoidably bad one would
/// overstate the speedup it is there to bound.
constexpr int kBins = 16;
constexpr int kMaxLeaf = 4;

struct BuildCtx {
    const std::vector<Triangle>* tris;
    std::vector<int>* idx;
    std::vector<BvhNode>* nodes;
    std::vector<Aabb> bounds;     // per triangle, computed once
    std::vector<Vec3> centroids;
    int max_depth = 0;
};

inline int build_recursive(BuildCtx& ctx, int first, int count, int depth = 1) {
    ctx.max_depth = std::max(ctx.max_depth, depth);
    const int node_index = static_cast<int>(ctx.nodes->size());
    ctx.nodes->push_back({});

    Aabb node_box, centroid_box;
    for (int i = 0; i < count; ++i) {
        const int t = (*ctx.idx)[first + i];
        node_box.expand(ctx.bounds[t]);
        centroid_box.expand(ctx.centroids[t]);
    }

    auto write = [&](int left_first, int cnt) {
        BvhNode& n = (*ctx.nodes)[node_index];
        n.lo = node_box.lo;
        n.hi = node_box.hi;
        n.left_first = left_first;
        n.count = cnt;
    };

    if (count <= kMaxLeaf) {
        write(first, count);
        return node_index;
    }

    // Split along the axis with the widest spread of centroids.
    const Vec3 e = centroid_box.extent();
    int axis = 0;
    if (e.y > e.x) axis = 1;
    if (e.z > (axis == 0 ? e.x : e.y)) axis = 2;

    const float lo = axis == 0 ? centroid_box.lo.x : (axis == 1 ? centroid_box.lo.y : centroid_box.lo.z);
    const float hi = axis == 0 ? centroid_box.hi.x : (axis == 1 ? centroid_box.hi.y : centroid_box.hi.z);

    if (hi - lo < 1e-8f) {           // degenerate: every centroid coincident
        write(first, count);
        return node_index;
    }

    const float scale = kBins / (hi - lo);
    auto bin_of = [&](int t) {
        const Vec3& c = ctx.centroids[t];
        const float v = axis == 0 ? c.x : (axis == 1 ? c.y : c.z);
        return std::min(kBins - 1, static_cast<int>((v - lo) * scale));
    };

    Aabb bin_box[kBins];
    int  bin_count[kBins] = {0};
    for (int i = 0; i < count; ++i) {
        const int t = (*ctx.idx)[first + i];
        const int b = bin_of(t);
        bin_box[b].expand(ctx.bounds[t]);
        bin_count[b]++;
    }

    // Sweep both directions once so every candidate split is costed in O(bins).
    float left_area[kBins - 1], right_area[kBins - 1];
    int   left_n[kBins - 1], right_n[kBins - 1];
    {
        Aabb acc; int n = 0;
        for (int i = 0; i < kBins - 1; ++i) {
            acc.expand(bin_box[i]); n += bin_count[i];
            left_area[i] = acc.half_area(); left_n[i] = n;
        }
    }
    {
        Aabb acc; int n = 0;
        for (int i = kBins - 1; i > 0; --i) {
            acc.expand(bin_box[i]); n += bin_count[i];
            right_area[i - 1] = acc.half_area(); right_n[i - 1] = n;
        }
    }

    float best_cost = std::numeric_limits<float>::max();
    int   best_split = -1;
    for (int i = 0; i < kBins - 1; ++i) {
        if (left_n[i] == 0 || right_n[i] == 0) continue;
        const float cost = left_area[i] * left_n[i] + right_area[i] * right_n[i];
        if (cost < best_cost) { best_cost = cost; best_split = i; }
    }

    // A leaf that costs less than any split is a leaf. node_box.half_area() * count is
    // the cost of testing every triangle here instead of descending.
    if (best_split < 0 || best_cost >= node_box.half_area() * count) {
        write(first, count);
        return node_index;
    }

    const auto begin = ctx.idx->begin() + first;
    const auto mid = std::partition(begin, begin + count,
                                    [&](int t) { return bin_of(t) <= best_split; });
    const int left_count = static_cast<int>(mid - begin);

    if (left_count == 0 || left_count == count) {   // partition made no progress
        write(first, count);
        return node_index;
    }

    build_recursive(ctx, first, left_count, depth + 1);         // left is implicit: node+1
    const int right = build_recursive(ctx, first + left_count, count - left_count, depth + 1);
    write(right, 0);
    return node_index;
}

}  // namespace detail

inline void Scene::build() {
    const int n = static_cast<int>(triangles.size());
    indices.resize(n);
    for (int i = 0; i < n; ++i) indices[i] = i;

    detail::BuildCtx ctx;
    ctx.tris = &triangles;
    ctx.idx = &indices;
    ctx.nodes = &nodes;
    ctx.bounds.resize(n);
    ctx.centroids.resize(n);
    for (int i = 0; i < n; ++i) {
        ctx.bounds[i] = detail::tri_bounds(triangles[i]);
        ctx.centroids[i] = ctx.bounds[i].centroid();
    }

    nodes.clear();
    nodes.reserve(2 * n);
    if (n > 0) detail::build_recursive(ctx, 0, n);
    max_depth = ctx.max_depth;
}

}  // namespace argos
