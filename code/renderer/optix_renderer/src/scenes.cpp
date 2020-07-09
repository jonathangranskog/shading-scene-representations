#include "scenes.hpp"
#include "util.hpp"
#include <algorithm>
#include <utility>
#include <memory>
#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>
#include <filesystem/path.h>
#include "material.h"
#include "texture.hpp"
#include "json_util.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace optix;

GenericScene::~GenericScene() {
    // Get context
    Context context = lightBuffer->getContext();
    for (auto& mat : objectMaterials) {
        if (mat.textureIndex != -1) {
            TextureSampler sampler = context->getTextureSamplerFromId(mat.textureIndex);
            sampler->getBuffer()->destroy();
            sampler->destroy();
            mat.textureIndex = -1;
        }
    }
    
    objects.clear();
    objectMaterials.clear();
    lights.clear();
    lightSources.clear();
    lightBuffer->destroy();
    geometry.clear();
    matToIndex.clear();
}

// This function loads a json string and constructs the appropriate objects
void GenericScene::loadJson(const std::string& jsonString, Context& context, Material material) {
    rapidjson::Document jsonDocument;
    jsonDocument.Parse(jsonString.c_str());
    
    loadMaterials(jsonDocument, context);
    loadGeometry(jsonDocument, context, material);
    loadLights(jsonDocument, context, material);
    loadCamera(jsonDocument, context);
    
    // Read other variables
    if (jsonDocument.HasMember("bounces")) context["maxBounces"]->setUint(readInt(jsonDocument["bounces"]));
    if (jsonDocument.HasMember("hide_lights")) context["hideLights"]->setInt(readBoolean(jsonDocument["hide_lights"]));

    if (jsonDocument.HasMember("top_level_acceleration")) {
        topLevelAcceleration = readString(jsonDocument["top_level_acceleration"]);
    } else {
        topLevelAcceleration = "NoAccel";
    }

    // Find and set max material id
    int maxId = 1;
    for (auto& mat : objectMaterials) maxId = max(mat.id + 1, maxId);
    context["maxId"]->setInt(maxId);
}

void GenericScene::loadJsonFile(const std::string& jsonFile, Context context, Material material) {
    std::ifstream in(jsonFile.c_str());
    std::stringstream sstr;
    sstr << in.rdbuf();
    std::string jsonString = sstr.str();
    loadJson(jsonString, context, material);
}

void GenericScene::loadCamera(rapidjson::Document& doc, Context context) {
    assert(doc.HasMember("camera"));
    const rapidjson::Value& jsonCam = doc["camera"];

    float3 position = readFloat3(jsonCam["position"]);
    float3 lookAt = readFloat3(jsonCam["lookat"]);

    float near = readFloat(jsonCam["near"]);
    float far = readFloat(jsonCam["far"]);
    float fov = readFloat(jsonCam["fov"]);

    // Set up camera
    viewMatrix = createLookAt(position,
                              lookAt,
                              make_float3(0, 1, 0));

    perspMatrix = createPerspective(fov, 1.f, near, far);

    context["projection"]->setMatrix4x4fv(false, perspMatrix.inverse().getData());
    context["view"]->setMatrix4x4fv(false, viewMatrix.inverse().getData());
}

void GenericScene::setCamera(float px, float py, float pz, float lx, float ly, float lz, Context context) {
    float3 position = make_float3(px, py, pz);
    float3 lookAt = make_float3(lx, ly, lz);
    viewMatrix = createLookAt(position, lookAt, make_float3(0, 1, 0));

    context["view"]->setMatrix4x4fv(false, viewMatrix.inverse().getData());
}

void GenericScene::loadMaterials(rapidjson::Document& doc, Context context) {
    if (!doc.HasMember("materials")) return;

    const rapidjson::Value& jsonMats = doc["materials"];
    int index = 0;

    for (auto& jsonMat : jsonMats.GetObject()) {
        const rapidjson::Value& color        = jsonMat.value["color"];
        const rapidjson::Value& emission     = jsonMat.value["emission"];
        const rapidjson::Value& roughness    = jsonMat.value["roughness"];
        const rapidjson::Value& ior          = jsonMat.value["ior"];
        const rapidjson::Value& texture      = jsonMat.value["texture"];
        const rapidjson::Value& texture_freq = jsonMat.value["texture_frequency"];
        const rapidjson::Value& id           = jsonMat.value["id"];
        
        SurfaceMat m;
        m.color = readFloat3(color);
        m.emission = readFloat3(emission);
        m.roughness = readFloat(roughness);
        m.ior = readFloat(ior);

        float2 freq = readFloat2(texture_freq);
        m.freqx = freq.x;
        m.freqy = freq.y;
        m.lightIndex = -1;
        m.id = readInt(id);
    
        std::string tex = readString(texture);
        if (tex == "") {
            m.textureIndex = -1;
        } else {
            TextureSampler sampler = loadTexture(tex, context);
            m.textureIndex = sampler->getId();
        }

        matToIndex[jsonMat.name.GetString()] = index;
        objectMaterials.push_back(m);
        index++;
    }
}

void GenericScene::loadGeometry(rapidjson::Document& doc, Context& context, Material material) {
    if (!doc.HasMember("geometry")) return;

    const rapidjson::Value& arr = doc["geometry"];
    assert(arr.IsArray());

    for (rapidjson::SizeType i = 0; i < arr.Size(); i++) {
        const rapidjson::Value& obj = arr[i];
        std::string type = readString(obj["type"]);
        SceneObjPtr ptr;
        if (type == "grid") {
            float3 position = readFloat3(obj["position"]);
            float3 normal = readFloat3(obj["normal"]);
            float3 size = readFloat3(obj["size"]);
            ptr = createGrid(position, normal, size, context, material);
        } else if (type == "sphere") {
            float3 position = readFloat3(obj["position"]);
            float radius = readFloat(obj["radius"]);
            ptr = createSphere(context, material, position, radius);
        } else if (type == "file") {
            std::string filename = readString(obj["filename"]);
            ptr = loadObj(filename, context, material)[0];
        } else {
            std::cout << "Unrecognized object type: " << type << std::endl;
            continue;
        }

        Matrix4x4 xform = readMat4(obj["transform"]);
        SurfaceMat& mat = objectMaterials[matToIndex[readString(obj["material"])]];
        ptr->instance["mat"]->setUserData(sizeof(SurfaceMat), (void*)&mat);
        ptr->transform->setMatrix(false, xform.getData(), xform.inverse().getData());

        objects.push_back(ptr);
        geometry.push_back(ptr->transform);
    }
}

void GenericScene::loadLights(rapidjson::Document& doc, Context context, Material material) {
    if (!doc.HasMember("lights")) return;
    
    const rapidjson::Value& arr = doc["lights"];
    assert(arr.IsArray());

    for (rapidjson::SizeType i = 0; i < arr.Size(); i++) {
        const rapidjson::Value& obj = arr[i];
        std::string type = readString(obj["type"]);
        SceneObjPtr ptr;
        LightSource src;
        if (type == "grid") {
            float3 position = readFloat3(obj["position"]);
            float3 normal = readFloat3(obj["normal"]);
            float3 size = readFloat3(obj["size"]);
            float sz = fmaxf(size);
            size = make_float3(sz);
            ptr = createGrid(position, normal, size, context, material);
            src.type = 0;
            src.normal = normal;
            src.size = sz;
            src.center = position;
        } else if (type == "sphere") {
            float3 position = readFloat3(obj["position"]);
            float radius = readFloat(obj["radius"]);
            ptr = createSphere(context, material, position, radius);
            src.type = 1;
            src.size = radius;
            src.center = position;
            src.normal = make_float3(0, -1, 0);
        } else {
            std::cout << "Unrecognized light type: " << type << std::endl;
            continue;
        }

        Matrix4x4 xform = readMat4(obj["transform"]);
        SurfaceMat& mat = objectMaterials[matToIndex[readString(obj["material"])]];
        src.emission = mat.emission;
        mat.lightIndex = (int)i;
        ptr->instance["mat"]->setUserData(sizeof(SurfaceMat), (void*)&mat);
        ptr->transform->setMatrix(false, xform.getData(), xform.inverse().getData());
        src.transform = xform;

        lightSources.push_back(src);
        lights.push_back(ptr);
        geometry.push_back(ptr->transform);
    }

    lightBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    lightBuffer->setElementSize(sizeof(LightSource));
    lightBuffer->setSize(lightSources.size());

    memcpy((void*)lightBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), lightSources.data(), sizeof(LightSource) * lightSources.size());
    lightBuffer->unmap();

    context["lights"]->setBuffer(lightBuffer);
    context["numLights"]->setUint(lightSources.size());
}

void GenericScene::initAccel(Context context) {
    // Set acceleration structure for top level
    topObject = context->createGroup(geometry.begin(), geometry.end());
    topObject->setAcceleration(context->createAcceleration(topLevelAcceleration));
    context["topObject"]->set(topObject);
}
