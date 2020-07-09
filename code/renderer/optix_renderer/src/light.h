#ifndef LIGHT_H
#define LIGHT_H

#include "kernel_util.h"
#include <optix_world.h>
#include "definitions.h"
#include "sampling.h"

using namespace optix;

// Struct defining a spherical light or a quad light
struct LightSource {
    Matrix4x4 transform;
    float3 emission;
    float3 center;
    float3 normal;
    float size;
    int type;
};

struct LightInteraction {
    float3 o;
    float3 n;
    float3 ln;
    float3 wi;
    float pdf;
    float dist;
};

// Direct sampling of a quad area light (area-based)
RT_FUNCTION float3 sampleQuadLight(const float2& sample, LightInteraction& li, const LightSource& light) {
    Basis basis = createBasis(light.normal);
    float3 p = light.center + basis.x * (sample.x * 2 * light.size - light.size) + basis.y * (sample.y * 2 * light.size - light.size);
    float4 pw = light.transform * make_float4(p, 1.f);
    p = make_float3(pw.x, pw.y, pw.z);
    float4 ln = light.transform * make_float4(light.normal, 0);
    li.ln = make_float3(ln.x, ln.y, ln.z);
    li.wi = normalize(p - li.o);
    li.pdf = 1.f / (4 * light.size * light.size);
    li.pdf *= dot(p - li.o, p - li.o);
    li.pdf /= abs(dot(li.wi, li.ln));
    li.dist = length(p - li.o);
    return light.emission / (4 * light.size * light.size) / li.pdf;
}

// Direct sampling of a sphere light (solid angle based)
RT_FUNCTION float3 sampleSphereLight(const float2& sample, LightInteraction& li, const LightSource& light) {
    float4 cw = light.transform * make_float4(light.center, 1);
    float3 v = make_float3(cw.x, cw.y, cw.z) - li.o;
    float3 diff = normalize(v);
    Basis basis = createBasis(diff);
    float tmp = light.size / length(v);
    float cosThetaMax = sqrtf(1 - tmp * tmp);

    float3 dir = squareToSphereCap(sample, cosThetaMax);
    li.pdf = sphereCapPdf(dir, cosThetaMax);
    dir = fromLocalSpace(dir, basis);
    float3 d = sqrt(abs(dot(v, v) - light.size * light.size)) * dir;
    li.ln = normalize(d - v);
    li.wi = dir;

    // Compute intersection distance
    float3 o = v;
    float3 dd = normalize(li.wi);
    float a = dot(dd, dd);
    float b = - 2 * dot(o, dd);
    float c = dot(o, o) - light.size * light.size;
    float disc = b * b - 4 * a * c;
    disc = sqrtf(disc);
    float t = 0.f;
    if (disc >= 0.f) {
        float tmin = (-b - disc) / (2 * a);
        float tmax = (-b + disc) / (2 * a);
        t = tmin;
        if (tmin <= tmax) {
            t = (tmin < 0) ? tmax : tmin;
        }   
    }

    li.dist = t;

    // Emission is not affected by area
    return light.emission  / (4 * M_PIf * light.size * light.size) / li.pdf;
}

// Return emission and direction of light
RT_FUNCTION float3 sampleLight(const float2& sample, LightInteraction& li, const LightSource& light) {
    if (light.type == 0) {
        return sampleQuadLight(sample, li, light);
    } else {
        return sampleSphereLight(sample, li, light);
    }
}

// Evaluate quad light emission
RT_FUNCTION float3 evalQuadLight(LightInteraction& li, const LightSource& light) {
    float4 ln = light.transform * make_float4(light.normal, 0);
    li.ln = make_float3(ln.x, ln.y, ln.z);
    li.pdf = 1.f / (4 * light.size * light.size);
    li.pdf *= li.dist * li.dist;
    li.pdf /= abs(dot(li.wi, li.ln));
    return light.emission / (4 * light.size * light.size);
}

// Evaluate sphere light emission
RT_FUNCTION float3 evalSphereLight(LightInteraction& li, const LightSource& light) {
    float4 cw = light.transform * make_float4(light.center, 1);
    float3 v = make_float3(cw.x, cw.y, cw.z) - li.o;
    float3 diff = normalize(v);
    Basis basis = createBasis(diff);
    float tmp = light.size / length(v);
    float cosThetaMax = sqrtf(1 - tmp * tmp);

    li.pdf = sphereCapPdf(toLocalSpace(li.wi, basis), cosThetaMax);
    float3 d = sqrt(dot(v, v) - light.size * light.size) * li.wi;
    li.ln = normalize(d - v);
    return light.emission / (4 * M_PIf * light.size * light.size);
}

RT_FUNCTION float3 evalLight(LightInteraction& li, const LightSource& light) {
    if (light.type == 0) {
        return evalQuadLight(li, light);
    } else {
        return evalSphereLight(li, light);
    }
}



#endif


