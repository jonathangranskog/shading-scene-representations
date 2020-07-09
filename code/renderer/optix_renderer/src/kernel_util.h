#ifndef KERNEL_UTIL_H
#define KERNEL_UTIL_H

#include <optix_world.h>
#include "definitions.h"

using namespace optix;

struct VertexAttributes {
    float3 normal;
    float3 position;
    float2 uv;
};

// Normal ray payload
struct Payload {
    float3 result;
    float3 diffuseResult;
    float3 attenuation;
    float3 diffuseAtten;
    float3 normal;
    float3 albedo;
    float3 direct;
    float3 pos;
    float3 dir;
    float id;
    float depth;
    float roughness;
    float bsdfPdf;
    unsigned bounce;
    unsigned seed;
    bool stop;
};

// Shadow ray payload
struct ShadowPayload {
    bool hit;
};

// Closest hit ray payload
// Finds closest hit without computing shading
struct ClosestPayload {
    bool hit;
    float3 pos;
    float3 normal;
};

struct Basis {
    float3 x;
    float3 y;
    float3 z;
};

// Creates a basis based on a single vector that can be used to transform between coordinate systems
RT_FUNCTION Basis createBasis(const float3& n) {
    // http://jcgt.org/published/0006/01/01/
    Basis basis;
    float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    basis.x = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    basis.y = make_float3(b, sign + n.y * n.y * a, -n.y);
    basis.z = n;
    return basis;
}

// Converts a vector to the local space of the basis
RT_FUNCTION float3 toLocalSpace(const float3& v, const Basis& basis) {
    return make_float3(dot(basis.x, v), dot(basis.y, v), dot(basis.z, v));
}

// Converts a vector from the local space of the basis
RT_FUNCTION float3 fromLocalSpace(const float3& v, const Basis& basis) {
    return v.x * basis.x + v.y * basis.y + v.z * basis.z;
}

RT_FUNCTION float balanceHeuristic(float pdf1, float pdf2) {
    return pdf1 / (pdf1 + pdf2);
}


#endif