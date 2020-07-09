#include "geometry.hpp"
#include "programs.hpp"
#include "tinyobj/tiny_obj_loader.h"
#include "random.h"
#include "kernel_util.h"

using namespace optix;


// Create any scene object and set its material and intersection program
SceneObject::SceneObject(Context& context, Material material, Program intersect, Program bounds) {
    geometry = context->createGeometry();
    geometry->setIntersectionProgram(intersect);
    geometry->setBoundingBoxProgram(bounds);

    instance = context->createGeometryInstance();
    instance->setMaterialCount(1);
    instance->setMaterial(0, material);
    instance->setGeometry(geometry);

    group = context->createGeometryGroup();
    group->addChild(instance);
    group->setAcceleration(context->createAcceleration("Bvh"));

    transform = context->createTransform();
    Matrix4x4 identity = optix::Matrix4x4::identity();
    transform->setMatrix(false, identity.getData(), identity.getData());
    transform->setChild(group);
}

SceneObject::~SceneObject() {
    group->getAcceleration()->destroy();
    group->destroy();
    instance->destroy();
    geometry->destroy();
    if (isTriMesh) {
        attributeBuffer->destroy();
        indexBuffer->destroy();
    }
}

// Creates a triangle mesh based on vertices, normals and indices
SceneObjPtr createTriangleMesh(Context context, Material material, const std::vector<uint3>& idx, const std::vector<float3>& p, const std::vector<float3>& n, const std::vector<float2>& uv) {
    SceneObjPtr obj = SceneObjPtr(new SceneObject(context, material, Programs::find("triangleIntersect"), Programs::find("triangleBounds")));
    obj->isTriMesh = true;

    std::vector<VertexAttributes> vertices(p.size());
    for (int i = 0; i < vertices.size(); i++) {
        vertices[i].position = p[i];
        vertices[i].normal = n[i];
        vertices[i].uv = uv[i];
    }

    obj->indexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3);
    obj->attributeBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    obj->attributeBuffer->setElementSize(sizeof(VertexAttributes));

    obj->indexBuffer->setSize(idx.size());
    obj->attributeBuffer->setSize(vertices.size());

    // Move to Optix Buffers
    std::memcpy(obj->indexBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), idx.data(), idx.size() * sizeof(uint3));
    std::memcpy(obj->attributeBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), vertices.data(), vertices.size() * sizeof(VertexAttributes));

    obj->indexBuffer->unmap();
    obj->attributeBuffer->unmap();

    // Set geometry buffers
    obj->geometry["indexBuffer"]->setBuffer(obj->indexBuffer);
    obj->geometry["attributeBuffer"]->setBuffer(obj->attributeBuffer);
    obj->geometry->setPrimitiveCount(idx.size());
    
    return obj;
}

// Load an OBJ file using tinyobj, must be a triangle mesh with normals
std::vector<SceneObjPtr> loadObj(const std::string& path, Context context, Material material) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warning, error;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warning, &error, path.c_str());

    if (!error.empty()) {
        std::cout << "ERROR (TINYOBJ): " << error << std::endl;
    }

    if (!ret) {
        std::cout << "Failed to load file: " << path << std::endl;
        return std::vector<SceneObjPtr>();
    }

    std::vector<SceneObjPtr> objects = std::vector<SceneObjPtr>();

    for (int i = 0; i < shapes.size(); i++) {
        const tinyobj::mesh_t& mesh = shapes[i].mesh;
        
        std::vector<uint3> indices = std::vector<uint3>();
        std::vector<float3> positions = std::vector<float3>();
        std::vector<float3> normals = std::vector<float3>();
        std::vector<float2> uvs = std::vector<float2>();

        size_t index_offset = 0;
        for (size_t f = 0; f < mesh.num_face_vertices.size(); f++) {
            size_t fnum = shapes[i].mesh.num_face_vertices[f];

            int px = mesh.indices[index_offset + 0].vertex_index;
            int py = mesh.indices[index_offset + 1].vertex_index;
            int pz = mesh.indices[index_offset + 2].vertex_index;
            int nx = mesh.indices[index_offset + 0].normal_index;
            int ny = mesh.indices[index_offset + 1].normal_index;
            int nz = mesh.indices[index_offset + 2].normal_index;
            int tx = mesh.indices[index_offset + 0].texcoord_index;
            int ty = mesh.indices[index_offset + 1].texcoord_index;
            int tz = mesh.indices[index_offset + 2].texcoord_index;

            float3 p0 = make_float3(static_cast<float>(attrib.vertices[3 * px + 0]),
                                    static_cast<float>(attrib.vertices[3 * px + 1]),
                                    static_cast<float>(attrib.vertices[3 * px + 2]));

            float3 n0 = make_float3(static_cast<float>(attrib.normals[3 * nx + 0]),
                                    static_cast<float>(attrib.normals[3 * nx + 1]),
                                    static_cast<float>(attrib.normals[3 * nx + 2]));

            float2 uv0 = make_float2(0.f);
            if (tx != -1) {
                uv0.x = static_cast<float>(attrib.texcoords[2 * tx + 0]);
                uv0.y = static_cast<float>(attrib.texcoords[2 * tx + 1]);
            }

            float3 p1 = make_float3(static_cast<float>(attrib.vertices[3 * py + 0]),
                                    static_cast<float>(attrib.vertices[3 * py + 1]),
                                    static_cast<float>(attrib.vertices[3 * py + 2]));

            float3 n1 = make_float3(static_cast<float>(attrib.normals[3 * ny + 0]),
                                    static_cast<float>(attrib.normals[3 * ny + 1]),
                                    static_cast<float>(attrib.normals[3 * ny + 2]));
            
            float2 uv1 = make_float2(0.f);
            if (ty != -1) {
                uv1.x = static_cast<float>(attrib.texcoords[2 * ty + 0]);
                uv1.y = static_cast<float>(attrib.texcoords[2 * ty + 1]);
            }
            

            float3 p2 = make_float3(static_cast<float>(attrib.vertices[3 * pz + 0]),
                                    static_cast<float>(attrib.vertices[3 * pz + 1]),
                                    static_cast<float>(attrib.vertices[3 * pz + 2]));

            float3 n2 = make_float3(static_cast<float>(attrib.normals[3 * nz + 0]),
                                    static_cast<float>(attrib.normals[3 * nz + 1]),
                                    static_cast<float>(attrib.normals[3 * nz + 2]));

            float2 uv2 = make_float2(0.f);
            if (tz != -1) {
                uv2.x = static_cast<float>(attrib.texcoords[2 * tz + 0]);
                uv2.y = static_cast<float>(attrib.texcoords[2 * tz + 1]);
            }

            positions.push_back(p0);
            normals.push_back(n0);
            uvs.push_back(uv0);
            unsigned x = positions.size() - 1;

            positions.push_back(p1);
            normals.push_back(n1);
            uvs.push_back(uv1);
            unsigned y = positions.size() - 1;

            positions.push_back(p2);
            normals.push_back(n2);
            uvs.push_back(uv2);
            unsigned z = positions.size() - 1;

            indices.push_back(make_uint3(x, y, z));
            index_offset += fnum;
        }

        SceneObjPtr obj = createTriangleMesh(context, material, indices, positions, normals, uvs);
        objects.push_back(obj);
    }

    return objects;
}

SceneObjPtr createGrid(const float3& v0, const float3& v1, const float3& v2, const float3& v3,
                       Context context, Material material) {
    float3 normal = normalize(-cross(v1 - v0, v2 - v0));
    createGrid(v0, v1, v2, v3, normal, context, material);
}

// Create a quad based on its vertices
SceneObjPtr createGrid(const float3& v0, const float3& v1, const float3& v2, const float3& v3,
                       const float3& normal, Context context, Material material) {

    std::vector<uint3> indices;
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> uvs;

    positions.push_back(v0);
    positions.push_back(v1);
    positions.push_back(v2);
    positions.push_back(v3);

    indices.push_back(optix::make_uint3(0, 1, 3));
    indices.push_back(optix::make_uint3(3, 1, 2));

    normals.push_back(normal);
    normals.push_back(normal);
    normals.push_back(normal);
    normals.push_back(normal);

    float2 uv0 = make_float2(0, 0);
    float2 uv1 = make_float2(0, 1);
    float2 uv2 = make_float2(1, 1);
    float2 uv3 = make_float2(1, 0);

    uvs.push_back(uv0);
    uvs.push_back(uv1);
    uvs.push_back(uv2);
    uvs.push_back(uv3);

    return createTriangleMesh(context, material, indices, positions, normals, uvs);
}

SceneObjPtr createGrid(const float3& pos, const float3& n, const float3& size, Context context, Material material) {
    float3 v0, v1, v2, v3;

    Basis basis = createBasis(n);
    float3 x = 0.5f * basis.x * size;
    float3 z = 0.5f * basis.y * size;

    v3 = pos - x - z;
    v2 = pos + x - z;
    v1 = pos + x + z;
    v0 = pos - x + z;

    return createGrid(v0, v1, v2, v3, n, context, material);
}

// Create an analytic sphere
SceneObjPtr createSphere(Context context, Material material, const float3& center, float radius) {
   SceneObjPtr obj =  SceneObjPtr(new SceneObject(context, material, Programs::find("sphereIntersect"), Programs::find("sphereBounds")));
   obj->geometry["center"]->setFloat(center);
   obj->geometry["radius"]->setFloat(radius);
   obj->geometry->setPrimitiveCount(1);
   return obj;
}