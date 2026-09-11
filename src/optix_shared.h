// =============================================================================
// QuBLAR -- the launch parameters, shared by the OptiX programs and their host
// =============================================================================
// One definition, included by both sides. A duplicated struct here is a silent
// corruption: the pipeline reads whatever bytes the host wrote, with no type check
// and no error, and the symptom is wrong geometry rather than a crash.
// =============================================================================

#pragma once

#include <optix.h>
#include "ray.hpp"

/// Geometry is passed as a flat vertex array rather than as the host's Triangle struct,
/// so the OptiX module does not have to include the BVH builder it exists to replace.
/// Three vertices per primitive, matching the buffer handed to optixAccelBuild, which
/// also guarantees the normal is computed from exactly the same numbers the hardware
/// intersected.
struct OptixParams {
    OptixTraversableHandle handle;
    const Ray*    rays;
    Hit*          hits;
    const float3* verts;       // 3 * n_triangles
    const int*    materials;   // n_triangles
    int           n_rays;
};
