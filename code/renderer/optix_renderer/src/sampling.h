#ifndef SAMPLING_H
#define SAMPLING_H

#include <optix_world.h>
#include "definitions.h"

// Sampling helper functions

using namespace optix;

RT_FUNCTION float2 squareToDisk(const float2& sample) {
    float r = sqrtf(sample.x);
    float theta = sample.y * M_PIf * 2;
    return make_float2(r * cosf(theta), r * sinf(theta));
}

RT_FUNCTION float3 squareToCosineHemisphere(const float2& sample) {
    float2 disk = squareToDisk(sample);
    return make_float3(disk.x, disk.y, sqrtf(1 - disk.x * disk.x - disk.y * disk.y));
}

RT_FUNCTION float cosineHemispherePdf(const float3& v) {
    if (v.z < 0) return 0;
    return v.z * M_1_PIf;
}

RT_FUNCTION float3 squareToGGX(const float2& sample, float alpha) {
    float a2 = alpha * alpha;
    float theta = acos(sqrt((1 - sample.x) / (sample.x * (a2 - 1) + 1)));
    float phi = 2 * M_PIf * sample.y;
    return make_float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
}

// https://hal.archives-ouvertes.fr/hal-01509746/document
RT_FUNCTION float3 squareToGGX2(const float2& sample, const float3& wi, float alpha) {
    // stretch view
    float3 V = normalize(make_float3(alpha * wi.x, alpha * wi.y, wi.z));
    // orthonormal basis
    float3 T1 = (V.z < 0.9999) ? normalize(cross(V, make_float3(0,0,1))) : make_float3(1,0,0);
    float3 T2 = cross(T1, V);
    // sample point with polar coordinates (r, phi)
    float a = 1.0 / (1.0 + V.z);
    float r = sqrt(sample.x);
    float phi = (sample.y<a) ? sample.y/a * M_PI : M_PI + (sample.y-a)/(1.0-a) * M_PI;
    float P1 = r*cos(phi);
    float P2 = r*sin(phi)*((sample.y<a) ? 1.0 : V.z);
    // compute normal
    float3 N = P1*T1 + P2*T2 + sqrt(max(0.0, 1.0 - P1*P1 - P2*P2))*V;
    // unstretch
    N = normalize(make_float3(alpha * N.x, alpha * N.y, max(0.0, N.z)));
    return N;
}

RT_FUNCTION float GGXPdf(const float3& wh, const float3& wi, float alpha) {
    float nh = wh.z;
    float a2 = alpha * alpha;
    float tmp = nh * nh * (a2 - 1) + 1;
    float D = a2 / (M_PIf * tmp * tmp);
    return nh * D / (4 * dot(wi, wh));
}

RT_FUNCTION float GGX2Pdf(const float3& wh, const float3& wi, float alpha) {
    // G1 * D / 4dot(N, wi)
    float a2 = alpha * alpha;
    float ni = wi.z;
    float tmp = sqrt(a2 + (1.f - a2) * ni * ni) + ni;
    float g = 2 * ni / tmp;

    float nh = wh.z;
    float tmp2 = nh * nh * (a2 - 1) + 1;
    float d = a2 / (M_PIf * tmp2 * tmp2);

    return abs(g * d / (4 * ni));
}

RT_FUNCTION float3 squareToCylinder(const float2& sample) {
    float wz = 2 * sample.x - 1;
    float theta = 2 * M_PIf * sample.y;
    float wx = cos(theta);
    float wy = sin(theta);
    return make_float3(wx, wy, wz);
}

RT_FUNCTION float3 squareToSphereCap(const float2& sample, float cosThetaMax) {
    float theta = acos(1 - sample.x + sample.x * cosThetaMax);
    float phi = M_PIf * 2 * sample.y;
    return make_float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
}

RT_FUNCTION float sphereCapPdf(const float3& v, float cosThetaMax) {
    float area = 2 * M_PIf * (1 - cosThetaMax);
    return 1.f / area;
}

#endif