#include "context.hpp"
#include "scenes.hpp"
#include "programs.hpp"
#include <filesystem/path.h>
#include <filesystem/resolver.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <ctime>

namespace py = pybind11;

void printVRAM(std::string prnt) {
    size_t freeByte ;
    size_t totalByte ;
    cudaMemGetInfo(&freeByte, &totalByte);
    std::cout << prnt << " " << freeByte << " " << totalByte << std::endl;
}

// C++ side interface to renderer
// Contains all OptiX programs and scene control

OptixContext::OptixContext(int sz, int device) {
    size = sz;
    ctx = optix::Context::create();
    ctx->setEntryPointCount(1);
    ctx->setRayTypeCount(3);
    ctx->setStackSize(1024);
    std::vector<int> devices;
    devices.push_back(device);
    ctx->setDevices(devices.begin(), devices.end());
    outputBuffer = ctx->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, size, size, 32);
    ctx["outputBuffer"]->set(outputBuffer);

    // Create all programs
    Programs::init(ctx);

    // Set ray generation program
    ctx->setRayGenerationProgram(0, Programs::find("raygen"));

    // Create material and set programs
    material = ctx->createMaterial();
    material->setClosestHitProgram(0, Programs::find("hit"));
    material->setClosestHitProgram(2, Programs::find("findClosest"));
    material->setAnyHitProgram(0, Programs::find("anyHit"));
    material->setAnyHitProgram(1, Programs::find("shadow"));
    material->setAnyHitProgram(2, Programs::find("findClosestAnyHit"));
    ctx->setMissProgram(0, Programs::find("miss"));

    // Initialize frame_seed 
    frameSeed = std::rand();

    // Default MIS active
    ctx["useMIS"]->setInt(1);
}

void OptixContext::loadSceneFile(std::string filename) {
    scene = std::unique_ptr<GenericScene>(new GenericScene());
    scene->loadJsonFile(filename, ctx, material);
    scene->initAccel(ctx);
}

void OptixContext::loadSceneJson(std::string desc) {
    scene = std::unique_ptr<GenericScene>(new GenericScene());
    scene->loadJson(desc, ctx, material);
    scene->initAccel(ctx);
}

optix::Matrix4x4 OptixContext::getViewMatrix() const {
    return scene->viewMatrix;
}

void OptixContext::setCamera(float px, float py, float pz, float lx, float ly, float lz) {
    if (scene == nullptr) return;
    scene->setCamera(px, py, pz, lx, ly, lz, ctx);
}

optix::Buffer OptixContext::getImage() const {
    return outputBuffer;
}

void OptixContext::setNEE(bool nee) {
    ctx["enableNEE"]->setInt(nee);
}

void OptixContext::render(int samples) {
    ctx["frame"]->setUint(frame);
    frameSeed = std::rand();
    ctx["frameSeed"]->setUint(frameSeed);
    frame++;
    for (int i = 0; i < samples; i++) {
        ctx["iteration"]->setUint(i);
        ctx->launch(0, size, size);
    }
}