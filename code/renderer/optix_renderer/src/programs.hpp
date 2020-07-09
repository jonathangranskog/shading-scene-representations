#pragma once

#include <map>
#include <string>
#include <cassert>
#include <optix_world.h>
#include <filesystem/path.h>
#include <filesystem/resolver.h>

using namespace optix;

// Storage of programs accessible from anywhere
// All static class

class Programs {
public:
    static std::map<std::string, Program> programs;

    static void createProgram(Context& ctx, const std::string& ptx, const std::string& name) {
        Program prog = ctx->createProgramFromPTXFile(ptx, name);
        programs.insert(std::pair<std::string, Program>(name, prog));
    }

    // Create all programs
    static void init(Context& ctx) {
        programs = std::map<std::string, Program>();
        filesystem::path kernelFolder = filesystem::path("renderer/optix_renderer/kernels/");
        filesystem::path renderFile = filesystem::path("render.ptx");
        filesystem::path intersectFile = filesystem::path("intersect.ptx");
        filesystem::path renderPtx = kernelFolder/renderFile;
        filesystem::path intersectPtx = kernelFolder/intersectFile;
        std::string renderPtxAbsolute = renderPtx.make_absolute().str();
        std::string intersectPtxAbsolute = intersectPtx.make_absolute().str();

        // Create ray generation program
        createProgram(ctx, renderPtxAbsolute, "raygen");

        // Create closest hit, miss and material
        createProgram(ctx, renderPtxAbsolute, "hit");
        createProgram(ctx, renderPtxAbsolute, "miss");
        createProgram(ctx, renderPtxAbsolute, "anyHit");
        createProgram(ctx, renderPtxAbsolute, "shadow");
        createProgram(ctx, renderPtxAbsolute, "findClosest");
        createProgram(ctx, renderPtxAbsolute, "findClosestAnyHit");

        // Create intersection programs
        createProgram(ctx, intersectPtxAbsolute, "triangleIntersect");
        createProgram(ctx, intersectPtxAbsolute, "triangleBounds");
        createProgram(ctx, intersectPtxAbsolute, "sphereIntersect");
        createProgram(ctx, intersectPtxAbsolute, "sphereBounds");
    }

    static Program find(const std::string& name) {
        auto it = programs.find(name);
        assert(it != programs.end());
        return it->second;
    }
};