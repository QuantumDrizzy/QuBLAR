// =============================================================================
// QuBLAR -- host side of the RT-core path
// =============================================================================
// ADR-002 item 3. Builds the hardware acceleration structure and the pipeline, then
// exposes one call with the same shape as the baseline kernel launch: rays in, hits
// out, on device memory the caller owns.
//
// Setup and traversal are kept strictly apart. Everything expensive and one-off --
// module compilation, pipeline linking, the acceleration build -- happens in init()
// and build(); trace() does nothing but launch. ADR-002 forbids folding build time
// into traversal time, and the cleanest way to honour that is to make it impossible.
// =============================================================================

#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "trace.cuh"
#include "optix_shared.h"

namespace argos {

#define OPTIX_CHECK(call)                                                             \
    do {                                                                              \
        OptixResult _r = (call);                                                      \
        if (_r != OPTIX_SUCCESS) {                                                    \
            std::fprintf(stderr, "OptiX error %d at %s:%d\n", (int)_r,                \
                         __FILE__, __LINE__);                                         \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

#define OPTIX_CHECK_LOG(call)                                                         \
    do {                                                                              \
        log_size = sizeof(log_buf);                                                   \
        OptixResult _r = (call);                                                      \
        if (_r != OPTIX_SUCCESS) {                                                    \
            std::fprintf(stderr, "OptiX error %d at %s:%d\n%s\n", (int)_r,            \
                         __FILE__, __LINE__, log_buf);                                \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
struct RecordData { int unused; };

class OptixTracer {
public:
    /// Milliseconds the acceleration structure took to build, and the bytes it occupies
    /// after compaction. Reported separately from any traversal number, never added.
    float  build_ms = 0.f;
    size_t gas_bytes = 0;

    void init(const std::string& module_path) {
        CUDA_CHECK(cudaFree(nullptr));                 // establish a context
        OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions opts = {};
        OPTIX_CHECK(optixDeviceContextCreate(0, &opts, &ctx_));

        optixDeviceContextGetProperty(ctx_, OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
                                      &rtcore_, sizeof(rtcore_));

        std::ifstream f(module_path, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "cannot open OptiX module: %s\n", module_path.c_str());
            std::exit(1);
        }
        std::stringstream ss; ss << f.rdbuf();
        const std::string ir = ss.str();

        OptixModuleCompileOptions mco = {};
        mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        pco_ = {};
        pco_.numPayloadValues   = 0;   // hit programs write the output buffer directly
        pco_.numAttributeValues = 2;   // built-in triangle barycentrics
        pco_.pipelineLaunchParamsVariableName = "params";
        pco_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pco_.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

        char log_buf[4096]; size_t log_size = sizeof(log_buf);
        OPTIX_CHECK_LOG(optixModuleCreate(ctx_, &mco, &pco_, ir.c_str(), ir.size(),
                                          log_buf, &log_size, &module_));

        OptixProgramGroupOptions pgo = {};
        OptixProgramGroupDesc rg = {}, ms = {}, ch = {};
        rg.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        rg.raygen.module = module_;
        rg.raygen.entryFunctionName = "__raygen__trace";
        ms.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        ms.miss.module = module_;
        ms.miss.entryFunctionName = "__miss__trace";
        ch.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        ch.hitgroup.moduleCH = module_;
        ch.hitgroup.entryFunctionNameCH = "__closesthit__trace";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(ctx_, &rg, 1, &pgo, log_buf, &log_size, &pg_rg_));
        OPTIX_CHECK_LOG(optixProgramGroupCreate(ctx_, &ms, 1, &pgo, log_buf, &log_size, &pg_ms_));
        OPTIX_CHECK_LOG(optixProgramGroupCreate(ctx_, &ch, 1, &pgo, log_buf, &log_size, &pg_ch_));

        OptixProgramGroup groups[3] = {pg_rg_, pg_ms_, pg_ch_};
        OptixPipelineLinkOptions plo = {};
        plo.maxTraceDepth = 1;
        OPTIX_CHECK_LOG(optixPipelineCreate(ctx_, &pco_, &plo, groups, 3,
                                            log_buf, &log_size, &pipeline_));

        build_sbt();
    }

    /// Upload the scene and build the hardware acceleration structure.
    ///
    /// Vertices go up in the scene's ORIGINAL triangle order, not the order the CPU BVH
    /// permuted them into. OptiX reports a primitive index into this buffer, and the
    /// baseline reports an index into the original array; uploading the permuted order
    /// would make the two tracers report different integers for the same surface, and
    /// the agreement check would be comparing labels rather than geometry.
    void build(const Scene& scene) {
        const int n_tris = static_cast<int>(scene.triangles.size());
        std::vector<float3> verts(3 * n_tris);
        std::vector<int>    mats(n_tris);
        for (int i = 0; i < n_tris; ++i) {
            const Triangle& t = scene.triangles[i];
            verts[3 * i + 0] = make_float3(t.v0.x, t.v0.y, t.v0.z);
            verts[3 * i + 1] = make_float3(t.v1.x, t.v1.y, t.v1.z);
            verts[3 * i + 2] = make_float3(t.v2.x, t.v2.y, t.v2.z);
            mats[i] = t.material;
        }

        CUDA_CHECK(cudaMalloc(&d_verts_, verts.size() * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&d_mats_, mats.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_verts_, verts.data(), verts.size() * sizeof(float3),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mats_, mats.data(), mats.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

        OptixBuildInput bi = {};
        CUdeviceptr vb = reinterpret_cast<CUdeviceptr>(d_verts_);
        unsigned flags[1] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
        bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        bi.triangleArray.vertexStrideInBytes = sizeof(float3);
        bi.triangleArray.numVertices = static_cast<unsigned>(verts.size());
        bi.triangleArray.vertexBuffers = &vb;
        bi.triangleArray.flags = flags;
        bi.triangleArray.numSbtRecords = 1;

        OptixAccelBuildOptions ao = {};
        ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
                      | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        ao.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes sizes = {};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx_, &ao, &bi, 1, &sizes));

        void* d_tmp = nullptr; void* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_tmp, sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(&d_out, sizes.outputSizeInBytes));

        unsigned long long* d_compacted = nullptr;
        CUDA_CHECK(cudaMalloc(&d_compacted, sizeof(unsigned long long)));
        OptixAccelEmitDesc emit = {};
        emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit.result = reinterpret_cast<CUdeviceptr>(d_compacted);

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        OPTIX_CHECK(optixAccelBuild(ctx_, 0, &ao, &bi, 1,
                                    reinterpret_cast<CUdeviceptr>(d_tmp), sizes.tempSizeInBytes,
                                    reinterpret_cast<CUdeviceptr>(d_out), sizes.outputSizeInBytes,
                                    &gas_, &emit, 1));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&build_ms, t0, t1));

        unsigned long long compacted = 0;
        CUDA_CHECK(cudaMemcpy(&compacted, d_compacted, sizeof(compacted), cudaMemcpyDeviceToHost));
        if (compacted > 0 && compacted < sizes.outputSizeInBytes) {
            CUDA_CHECK(cudaMalloc(&d_gas_, compacted));
            OPTIX_CHECK(optixAccelCompact(ctx_, 0, gas_,
                                          reinterpret_cast<CUdeviceptr>(d_gas_), compacted, &gas_));
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaFree(d_out);
            gas_bytes = static_cast<size_t>(compacted);
        } else {
            d_gas_ = d_out;
            gas_bytes = sizes.outputSizeInBytes;
        }

        cudaFree(d_tmp); cudaFree(d_compacted);
        cudaEventDestroy(t0); cudaEventDestroy(t1);

        if (!d_params_) CUDA_CHECK(cudaMalloc(&d_params_, sizeof(OptixParams)));
    }

    /// Launch. Device pointers in, device pointers out; nothing is allocated or copied
    /// here beyond the 40-byte parameter block, which the CUDA baseline pays for too in
    /// the form of its kernel arguments.
    void trace(const Ray* d_rays, Hit* d_hits, int n_rays) {
        OptixParams p{gas_, d_rays, d_hits, d_verts_, d_mats_, n_rays};
        CUDA_CHECK(cudaMemcpy(d_params_, &p, sizeof(p), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(pipeline_, 0,
                                reinterpret_cast<CUdeviceptr>(d_params_), sizeof(OptixParams),
                                &sbt_, n_rays, 1, 1));
    }

    unsigned rtcore_version() const { return rtcore_; }

    void free_scene() {
        cudaFree(d_verts_); cudaFree(d_mats_); cudaFree(d_gas_);
        d_verts_ = nullptr; d_mats_ = nullptr; d_gas_ = nullptr;
    }

private:
    void build_sbt() {
        SbtRecord<RecordData> rg{}, ms{}, ch{};
        OPTIX_CHECK(optixSbtRecordPackHeader(pg_rg_, &rg));
        OPTIX_CHECK(optixSbtRecordPackHeader(pg_ms_, &ms));
        OPTIX_CHECK(optixSbtRecordPackHeader(pg_ch_, &ch));

        void *d_rg = nullptr, *d_ms = nullptr, *d_ch = nullptr;
        CUDA_CHECK(cudaMalloc(&d_rg, sizeof(rg)));
        CUDA_CHECK(cudaMalloc(&d_ms, sizeof(ms)));
        CUDA_CHECK(cudaMalloc(&d_ch, sizeof(ch)));
        CUDA_CHECK(cudaMemcpy(d_rg, &rg, sizeof(rg), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ms, &ms, sizeof(ms), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ch, &ch, sizeof(ch), cudaMemcpyHostToDevice));

        sbt_ = {};
        sbt_.raygenRecord = reinterpret_cast<CUdeviceptr>(d_rg);
        sbt_.missRecordBase = reinterpret_cast<CUdeviceptr>(d_ms);
        sbt_.missRecordStrideInBytes = sizeof(ms);
        sbt_.missRecordCount = 1;
        sbt_.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(d_ch);
        sbt_.hitgroupRecordStrideInBytes = sizeof(ch);
        sbt_.hitgroupRecordCount = 1;
    }

    OptixDeviceContext ctx_ = nullptr;
    OptixModule        module_ = nullptr;
    OptixProgramGroup  pg_rg_ = nullptr, pg_ms_ = nullptr, pg_ch_ = nullptr;
    OptixPipeline      pipeline_ = nullptr;
    OptixPipelineCompileOptions pco_ = {};
    OptixShaderBindingTable sbt_ = {};
    OptixTraversableHandle gas_ = 0;

    float3* d_verts_ = nullptr;
    int*    d_mats_ = nullptr;
    void*   d_gas_ = nullptr;
    OptixParams* d_params_ = nullptr;
    unsigned rtcore_ = 0;
};

}  // namespace argos
