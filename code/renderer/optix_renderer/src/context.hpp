#pragma once

#include <optix_world.h>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <memory>
#include "scenes.hpp"

namespace py = pybind11;

class OptixContext {
private:
    optix::Buffer outputBuffer;
    optix::Context ctx;
    std::unique_ptr<GenericScene> scene;
    optix::Material material;
    int size;
    
    unsigned frame = 1;
    unsigned frameSeed = 1;

public:
    OptixContext(int size, int device);
    ~OptixContext() {}

    void loadSceneFile(std::string filename);
    void loadSceneJson(std::string desc);
    void setNEE(bool nee);
    void initAccel();

    void setCamera(float px, float py, float pz, float lx, float ly, float lz);

    optix::Matrix4x4 getViewMatrix() const;
    optix::Buffer getImage() const;
    void render(int spp);
};