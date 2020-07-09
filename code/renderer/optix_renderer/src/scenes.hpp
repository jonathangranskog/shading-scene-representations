#pragma once

#include <optix_world.h>
#include "random.h"
#include <vector>
#include <string>
#include <map>
#include "geometry.hpp"
#include "kernel_util.h"
#include "material.h"
#include "light.h"
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>

class GenericScene {
public:
    optix::Matrix4x4 viewMatrix;
    optix::Matrix4x4 perspMatrix;
    std::string topLevelAcceleration;
    float near, far, fov;
    std::vector<optix::Transform> geometry;
    optix::Group topObject;
    
    // Scene objects and materials
    std::vector<SceneObjPtr> objects;
    std::vector<SurfaceMat> objectMaterials;
    std::map<std::string, unsigned> matToIndex;

    // Light sources
    std::vector<LightSource> lightSources;
    std::vector<SceneObjPtr> lights;
    optix::Buffer lightBuffer;

    ~GenericScene();
    GenericScene() {}
    void loadJsonFile(const std::string& jsonFile, optix::Context context, optix::Material material);
    void loadJson(const std::string& jsonString, optix::Context& context, optix::Material material);
    void loadMaterials(rapidjson::Document& doc, optix::Context context);
    void loadGeometry(rapidjson::Document& doc, optix::Context& context, optix::Material material);
    void loadLights(rapidjson::Document& doc, optix::Context context, optix::Material material);
    void loadCamera(rapidjson::Document& doc, optix::Context context);

    void setCamera(float px, float py, float pz, float lx, float ly, float lz, optix::Context context);
    void initAccel(optix::Context context);
};
