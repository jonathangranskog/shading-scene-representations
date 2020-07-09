#pragma once

#include <optix_world.h>
#include <string>
#include <vector>
#include <memory>
#include <cstring>

using namespace optix;

// Class for all geometric objects

class SceneObject {
public:
    Transform transform;
    Geometry geometry;
    GeometryInstance instance;
    GeometryGroup group;
    unsigned id = 0;
    bool isTriMesh = false;

    Buffer indexBuffer;
    Buffer attributeBuffer;

    SceneObject(Context& context, Material material, Program intersect, Program bounds);
    ~SceneObject();
};

typedef std::shared_ptr<SceneObject> SceneObjPtr;

SceneObjPtr createTriangleMesh(Context context, Material material, const std::vector<uint3>& idx, const std::vector<float3>& p, const std::vector<float3>& n, const std::vector<float2>& uv);

std::vector<SceneObjPtr> loadObj(const std::string& path, Context context, Material material);

SceneObjPtr createGrid(const float3& pos, const float3& n, const float3& size, Context context, Material material);

SceneObjPtr createGrid(const float3& v0, const float3& v1, const float3& v2, const float3& v3, Context context, Material material);

SceneObjPtr createGrid(const float3& v0, const float3& v1, const float3& v2, const float3& v3, const float3& n, Context context, Material material);

SceneObjPtr createSphere(Context context, Material material, const float3& center, float radius);