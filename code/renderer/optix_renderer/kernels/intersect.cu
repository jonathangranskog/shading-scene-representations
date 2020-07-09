#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <kernel_util.h>

using namespace optix;

rtBuffer<VertexAttributes> attributeBuffer;
rtBuffer<uint3> indexBuffer;

rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, );
rtDeclareVariable(float3, shadingNormal, attribute shading_normal, );
rtDeclareVariable(float2, texCoord, attribute tex_coord, );

rtDeclareVariable(Ray, ray, rtCurrentRay, );

RT_PROGRAM void triangleIntersect(int primitiveIdx) 
{
    const uint3 indices = indexBuffer[primitiveIdx];

    VertexAttributes const& a0 = attributeBuffer[indices.x];
    VertexAttributes const& a1 = attributeBuffer[indices.y];
    VertexAttributes const& a2 = attributeBuffer[indices.z];

    const float3 v0 = a0.position;
    const float3 v1 = a1.position;
    const float3 v2 = a2.position;

    float3 n;
    float t, beta, gamma;

    if (intersect_triangle(ray, v0, v1, v2, n, t, beta, gamma))
    {
        if (rtPotentialIntersection(t))
        {
            const float alpha = 1.0f - beta - gamma;
            geometricNormal = n;
            shadingNormal = a0.normal * alpha + a1.normal * beta + a2.normal * gamma;
            texCoord = a0.uv * alpha + a1.uv * beta + a2.uv * gamma;
            
            rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void triangleBounds(int primitiveIdx, float result[6]) 
{
    const uint3 indices = indexBuffer[primitiveIdx];
    
    const float3 v0 = attributeBuffer[indices.x].position;
    const float3 v1 = attributeBuffer[indices.y].position;
    const float3 v2 = attributeBuffer[indices.z].position;

    const float area = optix::length(optix::cross(v1 - v0, v2 - v0));

    Aabb* aabb = (Aabb*) result;

    if (0.0f < area && !isinf(area))
    {
        aabb->m_min = fminf(fminf(v0, v1), v2);
        aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
    }
    else 
    {
        aabb->invalidate();
    }
}

rtDeclareVariable(float3, center, ,);
rtDeclareVariable(float, radius, ,);

RT_PROGRAM void sphereIntersect(int primIdx) {
    float3 o = center - ray.origin;
    float3 d = ray.direction;

    float a = dot(d, d);
    float b = -2 * dot(o, d);
    float c = dot(o, o) - radius * radius;
    float disc = b * b - 4 * a * c;
    
    if (disc >= 0.f) {
        disc = sqrtf(disc);
        float tmin = (-b - disc) / (2 * a);
        float tmax = (-b + disc) / (2 * a);
        if (tmin <= tmax) {
            float t = (tmin < 0) ? tmax : tmin;
            if (rtPotentialIntersection(t)) {
                float3 pt = ray.origin + t * ray.direction;
                float3 n = normalize(pt - center);
                geometricNormal = n;
                shadingNormal = n;
                texCoord = make_float2(0.f);
                rtReportIntersection(0);
            }     
        }
    }
}

RT_PROGRAM void sphereBounds(int primIdx, float result[6]) {
    Aabb* aabb = (Aabb*) result;
    if (radius > 0.f && !isinf(radius)) {
        aabb->m_min = center - make_float3(radius);
        aabb->m_max = center + make_float3(radius);
    } else {
        aabb->invalidate();
    }
}