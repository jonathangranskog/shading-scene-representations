#ifndef MATERIAL_H
#define MATERIAL_H

#include "kernel_util.h"
#include <optix_world.h>
#include "definitions.h"
#include "sampling.h"

#ifdef IS_CUDA
#include "optix_device.h"
#endif

using namespace optix;

#define FLT_EPS 1e-32

// Everything should be in local space
struct MaterialInteraction {
    float3 wi;
    float3 wo;
    float2 uv;
    float pdf;
};

// The parameters of the material
struct SurfaceMat {
    float3 color;
    float3 emission;
    float roughness;
    float ior;
    float freqx;
    float freqy;
    int visible;
    int lightIndex;
    int textureIndex;
    int id;
};

RT_FUNCTION SurfaceMat createMat(const float3& color, const float3& emission, int visible) {
    SurfaceMat mat;
    mat.color = color;
    mat.emission = emission;
    mat.visible = visible;
    mat.roughness = 1.0f;
    mat.ior = 15.f;
    mat.lightIndex = -1;
    mat.textureIndex = -1;
    mat.freqx = 1.f;
    mat.freqy = 1.f;
    mat.id = 0;
    return mat;
}

RT_FUNCTION float fresnel(float cti, float extIOR, float intIOR) {
    float etaI = extIOR, etaT = intIOR;
    if (extIOR == intIOR) return 0.f;
    if (cti < 0.f) {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
        cti = -cti;
    }
    float eta = etaI / etaT;
    float stt2 = eta * eta * (1 - cti * cti);
    if (stt2 > 1.f) return 1.f; // total internal reflection

    float ctt = sqrtf(1.f - stt2);
    float rs = (etaI * cti - etaT * ctt) / (etaI * cti + etaT * ctt);
    float rp = (etaT * cti - etaI * ctt) / (etaT * cti + etaI * ctt);
    return 0.5f * (rs * rs + rp * rp);
}

// Diffuse contribution (Lambert)
RT_FUNCTION float3 evalDiffuse(MaterialInteraction& mi, const SurfaceMat& mat) {
    mi.pdf = cosineHemispherePdf(mi.wo);

    float3 color = mat.color;

    #ifdef IS_CUDA
    if (mat.textureIndex != -1) {
        color = make_float3(rtTex2D<float4>(mat.textureIndex, mat.freqx * mi.uv.x, mat.freqy * mi.uv.y));
        color = mat.color * make_float3(fmaxf(color));
    }
    #endif

    return color / M_PIf;
}

// Specular contribution (GGX)
RT_FUNCTION float3 evalSpecular(MaterialInteraction& mi, const SurfaceMat& mat) {
    float3 wh = normalize(mi.wi + mi.wo);
    float alpha = mat.roughness * mat.roughness;
    float a2 = alpha * alpha;
    float nh = abs(wh.z);
    float ni = abs(mi.wi.z);
    float no = abs(mi.wo.z);
    float tmp = nh * nh * (a2 - 1) + 1;
    
    float D = a2 / (M_PIf * tmp * tmp);
    float F = fresnel(abs(dot(mi.wo, wh)), 1.f, mat.ior);

    float nom = 2 * ni * no;
    float denom1 = no * sqrt(a2 + (1 - a2) * ni * ni);
    float denom2 = ni * sqrt(a2 + (1 - a2) * no * no);
    float G = nom / (denom1 + denom2);

    float denom = 1.f / (4 * abs(mi.wi.z) * abs(mi.wo.z));

    float3 white = make_float3(1.f);

    mi.pdf = GGX2Pdf(wh, mi.wi, alpha);

    return abs(F * G * D * denom) * white;
    //return D * F * G * denom * white;
}

// Importance sample GGX lobe and return color result
RT_FUNCTION float3 sampleSpecular(const float2& sample, MaterialInteraction& mi, const SurfaceMat& mat) {
    float alpha = mat.roughness * mat.roughness;
    float3 wh = normalize(squareToGGX2(sample, mi.wi, alpha));
    if (wh.z < 0) wh = -wh;
    mi.wo = -mi.wi + 2 * dot(mi.wi, wh) * wh;
    float3 eval = evalSpecular(mi, mat);
    float3 result = eval / mi.pdf * abs(mi.wo.z);
    return result;
}

// Importance sample Lambert diffuse based on cosine hemisphere
RT_FUNCTION float3 sampleDiffuse(const float2& sample, MaterialInteraction& mi, const SurfaceMat& mat) {
    mi.wo = squareToCosineHemisphere(sample);
    mi.pdf = cosineHemispherePdf(mi.wo);
    float3 color = mat.color;

    #ifdef IS_CUDA
    if (mat.textureIndex != -1) {
        color = make_float3(rtTex2D<float4>(mat.textureIndex, mat.freqx * mi.uv.x, mat.freqy * mi.uv.y));
        color = mat.color * make_float3(fmaxf(color));
    }
    #endif

    return color;
}

// Samples an outgoing direction and returns the color divided by pdf
RT_FUNCTION float3 sampleMat(const float2& sample, MaterialInteraction& mi, const SurfaceMat& mat, int bounce, float3& diffuseAtten) {
    float spec = 1 - (mat.roughness - 0.05f) / (1.0f - 0.05f);
    float2 s = sample;

    if (s.x < spec) {
        s.x /= spec;
        diffuseAtten = make_float3(0.f);
        return sampleSpecular(s, mi, mat);
    } else {
        s.x = (s.x - spec) / (1 - spec);
        float3 result = sampleDiffuse(s, mi, mat);
        diffuseAtten = result;
        return result;
    }
}

// Both wi and wo should already be set for the material interaction
RT_FUNCTION float3 evalMat(MaterialInteraction& mi, const SurfaceMat& mat, int bounce, float3& diffuseEval) {
    float spec = 1 - (mat.roughness - 0.05f) / (1.0f - 0.05f);

    diffuseEval = (1 - spec) * evalDiffuse(mi, mat);
    float diffusePdf = mi.pdf;
    float3 specEval = spec * evalSpecular(mi, mat);
    float specPdf = mi.pdf;
    mi.pdf = spec * specPdf + (1 - spec) * diffusePdf;

    return specEval + diffuseEval;
}

#endif
