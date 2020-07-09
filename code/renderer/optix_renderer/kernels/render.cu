#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <random.h>
#include <sampling.h>
#include <light.h>
#include <kernel_util.h>
#include <material.h>

using namespace optix;

rtDeclareVariable(rtObject, topObject, ,);
rtDeclareVariable(unsigned, iteration, ,);
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );
rtDeclareVariable(Matrix4x4, view, ,);
rtDeclareVariable(Matrix4x4, projection, ,);
rtDeclareVariable(unsigned, seed, ,);
rtDeclareVariable(unsigned, frame, ,);
rtDeclareVariable(unsigned, frameSeed, ,);
rtDeclareVariable(unsigned, maxBounces, ,);

rtBuffer<LightSource> lights;
rtDeclareVariable(unsigned, numLights, ,);

rtBuffer<float, 3> outputBuffer;

RT_PROGRAM void raygen() {
    float3 color = make_float3(0.f);
    float3 normal = make_float3(0.f);
    float3 position = make_float3(0.f);
    float3 albedo = make_float3(0.f);
    float3 direct = make_float3(0.f);
    float3 diffuseComp = make_float3(0.f);
    float3 mirrorDir = make_float3(0.f);
    float3 secondHitPos = make_float3(0.f);
    float3 secondHitNormal = make_float3(0.f);
    float id = 0.f;
    float depth = 0.f;
    float roughness = 0.f;
    float shadowing = 0;
    float ao = 0;

    Payload payload;
    payload.seed = tea<8>(launchIndex.y * launchDim.x + launchIndex.x, frameSeed + iteration);

    payload.result = make_float3(0.f);
    payload.attenuation = make_float3(1.f);
    payload.diffuseAtten = make_float3(1.f);
    payload.diffuseResult = make_float3(0.f);
    payload.normal = make_float3(0.f);
    payload.pos = make_float3(0.f);
    payload.dir = make_float3(0.f, 1.f, 0.f);
    payload.direct = make_float3(0.f);
    payload.albedo = make_float3(0.f);
    payload.id = 0.f;
    payload.roughness = 0.f;
    payload.depth = 0.f;
    payload.bounce = 0;
    payload.stop = false;
    payload.bsdfPdf = 1.f;

    const float2 pixel = make_float2(launchIndex) + rng2(payload.seed);
    const float2 screen = make_float2(launchDim);
    const float2 ndc = (pixel / screen) * 2.f - 1.f;
    float4 imagePlanePos = make_float4(ndc.x, ndc.y, -1.f, 1.f);
    float4 p = projection * imagePlanePos;
    float4 origin = view * make_float4(0, 0, 0, 1.f);
    float4 d = view * make_float4(p.x, p.y, p.z, 1.f);
    float3 camOrig = make_float3(origin.x, origin.y, origin.z);
    float3 camDir = normalize(make_float3(d.x, d.y, d.z) - camOrig);

    payload.pos = camOrig;
    payload.dir = camDir;

    while (payload.bounce < maxBounces && !payload.stop) {
        Ray ray = make_Ray(payload.pos, payload.dir, 0, 1e-4f, RT_DEFAULT_MAX);
        rtTrace(topObject, ray, payload);

        // Store stuff into buffers
        if (payload.bounce == 0) {
            // Basic buffers
            normal += payload.normal;
            position += payload.pos;
            depth += payload.depth;
            albedo += payload.albedo;
            roughness += payload.roughness;
            id += payload.id;

            // Compute mirror buffer
            float3 tmpDir = normalize(camDir - 2 * dot(camDir, payload.normal) * payload.normal);
            Ray closestRay = make_Ray(payload.pos, tmpDir, 2, 1e-4f, RT_DEFAULT_MAX);
            mirrorDir += tmpDir;
            ClosestPayload closestPayload;
            closestPayload.hit = false;
            rtTrace(topObject, closestRay, closestPayload);
            if (closestPayload.hit) {
                secondHitNormal += closestPayload.normal;
                secondHitPos += closestPayload.pos;
            }

            // Compute direct shadowing buffer
            if (payload.stop) {
                // If hit light -- make white
                shadowing += 1.f;
            } else {
                // Loop through all lights and sample
                for (int selectedLight = 0; selectedLight < numLights; selectedLight++) {
                    LightInteraction liLight;
                    liLight.o = payload.pos;
                    liLight.n = payload.normal;
                    float3 lightSample = sampleLight(rng2(payload.seed), liLight, lights[selectedLight]);
                    
                    // Check that light is not facing away from us
                    float lndotl = dot(liLight.ln, -liLight.wi);
                    float ndotl = dot(liLight.n, liLight.wi);
                    if (lndotl > 0 && ndotl > 0) {
                        ShadowPayload shadowPayload;
                        shadowPayload.hit = false;
        
                        Ray shadowRay = make_Ray(payload.pos, liLight.wi, 1, 1e-4f, liLight.dist - 1e-4f);
                        rtTrace(topObject, shadowRay, shadowPayload);

                        if (!shadowPayload.hit) {
                            shadowing += 1.f / numLights;
                        }
                    }
                }
            }
            
            // Compute ambient occlusion buffer
            Basis basis = createBasis(payload.normal);
            float3 sample = squareToCosineHemisphere(rng2(payload.seed));
            float3 occlusionRayDir = fromLocalSpace(sample, basis);
            ShadowPayload occlusionPayload;
            occlusionPayload.hit = false;
            Ray occlusionRay = make_Ray(payload.pos, occlusionRayDir, 1, 1e-4f, 1.0f);
            rtTrace(topObject, occlusionRay, occlusionPayload);

            if (occlusionPayload.hit) {
                ao += 1.f;
            }
        }

        // Add to direct lighting buffer
        if (payload.bounce <= 1) {
            direct += payload.direct;    
        }

        payload.bounce++;
    }

    color += payload.result;
    diffuseComp += payload.diffuseResult;


    // Set buffer values (pretty ugly...)
    // maybe it would be faster to separate these into
    // their own buffers
    float weight = 1.f / float(iteration + 1);
    
    // beauty
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 0)] = weight * color.x + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 0)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 1)] = weight * color.y + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 1)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 2)] = weight * color.z + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 2)];

    // normal
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 3)] = weight * normal.x + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 3)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 4)] = weight * normal.y + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 4)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 5)] = weight * normal.z + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 5)];

    // depth
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 6)] = weight * depth + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 6)];

    // world position
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 7)] = weight * position.x + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 7)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 8)] = weight * position.y + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 8)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 9)] = weight * position.z + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 9)];

    // albedo
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 10)] = weight * albedo.x + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 10)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 11)] = weight * albedo.y + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 11)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 12)] = weight * albedo.z + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 12)];

    // roughness
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 13)] = weight * roughness + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 13)];

    // direct lighting
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 14)] = weight * direct.x + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 14)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 15)] = weight * direct.y + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 15)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 16)] = weight * direct.z + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 16)];

    // diffuse
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 17)] = weight * diffuseComp.x + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 17)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 18)] = weight * diffuseComp.y + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 18)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 19)] = weight * diffuseComp.z + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 19)];

    // mirror reflection direction
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 20)] = weight * mirrorDir.x + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 20)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 21)] = weight * mirrorDir.y + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 21)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 22)] = weight * mirrorDir.z + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 22)];

    // mirror intersection position
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 23)] = weight * secondHitPos.x + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 23)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 24)] = weight * secondHitPos.y + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 24)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 25)] = weight * secondHitPos.z + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 25)];

    // mirror intersection normal
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 26)] = weight * secondHitNormal.x + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 26)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 27)] = weight * secondHitNormal.y + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 27)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 28)] = weight * secondHitNormal.z + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 28)];
    
    // Shadowing and ambient occlusion
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 29)] = weight * shadowing + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 29)];
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 30)] = weight * (1 - ao) + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 30)];

    // object id buffer
    outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 31)] = weight * id + (1 - weight) * outputBuffer[make_uint3(launchIndex.x, launchIndex.y, 31)];
}

rtDeclareVariable(Payload, rayData, rtPayload, );
rtDeclareVariable(float3, shadingNormal, attribute shading_normal, );
rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, );
rtDeclareVariable(float2, uv, attribute tex_coord, );
rtDeclareVariable(float, intersectionDistance, rtIntersectionDistance, );
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(int, enableNEE, ,);
rtDeclareVariable(SurfaceMat, mat, ,);
rtDeclareVariable(int, maxId, ,);

RT_PROGRAM void hit() {
    // Set up normals
    float3 geoNormal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal));
    float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shadingNormal));
    
    // Fill in rayData for buffers
    rayData.depth = intersectionDistance;
    rayData.roughness = mat.roughness;
    rayData.albedo = mat.color;
    rayData.id = mat.id / float(maxId);
    rayData.pos = ray.origin + intersectionDistance * ray.direction;

    // Apply texture to albedo buffer
    if (mat.textureIndex != -1) {
        rayData.albedo = mat.color * make_float3(fmaxf(make_float3(rtTex2D<float4>(mat.textureIndex, mat.freqx * uv.x, mat.freqy * uv.y))));
    } else if (mat.roughness > 0.05f) {
        rayData.albedo = mat.color;
    } else {
        rayData.albedo = make_float3(0.f);
    }

    
    float3 contrib = make_float3(0.f);

    // If light source is hit directly
    if (dot(-ray.direction, normal) > 0) {
        if (fmaxf(mat.emission) > 0) {
            // Evaluate light contribution
            LightInteraction li;
            li.wi = ray.direction;
            li.o = ray.origin;
            li.n = rayData.normal;
            li.dist = length(rayData.pos - ray.origin);
            float3 eval = evalLight(li, lights[mat.lightIndex]);
            
            // NEE is disabled
            if (!enableNEE) {
                if (!isnan(li.pdf)) contrib += eval;
            }
            else {
                // NEE is enabled
                if (!isnan(li.pdf) && !isnan(rayData.bsdfPdf)) {
                    contrib += eval;
                    // Apply MIS weight if not direct hit from camera
                    if (rayData.bounce > 0)
                        contrib *= balanceHeuristic(rayData.bsdfPdf, li.pdf);
                }
            }

            rayData.stop = true;
        }
    } else {
        // Stop if we hit the backside of a light source
        contrib = make_float3(0.f);
        rayData.stop = true;
    }

    // Clamp to reduce fireflies, but don't clamp direct hits
    // This prevents us from clamping directly visible light sources
    if (rayData.bounce > 0) contrib = clamp(contrib, make_float3(0.f), make_float3(25.f));

    // Material sample
    Basis basis = createBasis(normal);
    MaterialInteraction mi;
    mi.uv = uv;
    mi.wi = toLocalSpace(-ray.direction, basis);
    
    float3 diffuseAtten = make_float3(0.f);
    float3 bsdfSample = sampleMat(rng2(rayData.seed), mi, mat, rayData.bounce, diffuseAtten);
    float3 worldWo = fromLocalSpace(mi.wo, basis);
    
    rayData.normal = normal;
    rayData.dir = worldWo;

    // Save into direct lighting buffer
    if (rayData.bounce <= 1) {
        rayData.direct += rayData.result + contrib * rayData.attenuation;
    }

    // Initialize diffuse buffer contribution
    float3 diffuseContrib = contrib;

    // If NEE is enabled and we are not outside our bounce budger
    // Sample area light using next event estimation
    if (!rayData.stop && enableNEE) {
        if (rayData.bounce < maxBounces - 1) {
            LightInteraction liLight;
            liLight.o = rayData.pos;
            liLight.n = rayData.normal;
            
            // Select light to sample
            int lightIndex = (int)(rng(rayData.seed) * numLights);
            float3 lightSample = sampleLight(rng2(rayData.seed), liLight, lights[lightIndex]) * numLights;

            // Make sure light is not facing away from us
            float lndotl = dot(-liLight.wi, liLight.ln);
            float ndotl = dot(liLight.wi, liLight.n);
            if (lndotl > 0 && ndotl > 0) {
                // Shadow ray
                ShadowPayload shadowPayload;
                shadowPayload.hit = false;
                
                Ray shadowRay = make_Ray(rayData.pos, liLight.wi, 1, 1e-4f, liLight.dist - 1e-4f);
                rtTrace(topObject, shadowRay, shadowPayload);

                // No occluding object
                if (!shadowPayload.hit) {

                    // Initialize data
                    MaterialInteraction miLight;
                    miLight.uv = uv;
                    miLight.wi = mi.wi;
                    miLight.wo = toLocalSpace(liLight.wi, basis);

                    // Compute diffuse contribution, overall contribution and MIS weight
                    float3 diffuseEval = make_float3(0.f);
                    float3 albedo = evalMat(miLight, mat, rayData.bounce, diffuseEval);
                    float weight = balanceHeuristic(liLight.pdf, miLight.pdf); // MIS weight
                    
                    // Add contributions if there are no NaNs to mess everything up
                    if (!isnan(liLight.pdf) && !isnan(miLight.pdf)) {
                        float3 tmp1 = (albedo * lightSample) * (weight * ndotl);
                        float3 tmp2 = (diffuseEval * lightSample) * weight * ndotl;
                        
                        // Clamp to prevent fireflies
                        diffuseContrib += clamp(tmp2, make_float3(0.f), make_float3(25.f));
                        contrib += clamp(tmp1, make_float3(0.f), make_float3(25.f));
                    }
                }
            }
        }
    }

    // Add in diffuse and overall results and update attenuations
    rayData.diffuseResult += rayData.diffuseAtten * diffuseContrib;
    rayData.result += rayData.attenuation * contrib;
    rayData.diffuseAtten *= diffuseAtten;
    rayData.attenuation *= bsdfSample;
    
    // Save bsdf pdf for MIS weight for possible direct light hit
    rayData.bsdfPdf = mi.pdf; 
}

rtDeclareVariable(int, hideLights, ,);

RT_PROGRAM void anyHit() {
    // Ignore hidden lights on direct camera hits
    if (hideLights && rayData.bounce == 0 && fmaxf(mat.emission) > 0) {
        rtIgnoreIntersection();
    } else if (mat.visible == 0) {
        rtIgnoreIntersection();
    }
}

rtDeclareVariable(ClosestPayload, closestPayload, rtPayload, );

// Used for computing AO
RT_PROGRAM void findClosest() {
    float3 geoNormal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal));
    float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shadingNormal));
    closestPayload.pos = ray.origin + intersectionDistance * ray.direction;
    closestPayload.normal = normal;
    closestPayload.hit = true;
}

RT_PROGRAM void findClosestAnyHit() {
    if (mat.visible == 0 || (fmaxf(mat.emission) > 0 && hideLights)) {
        rtIgnoreIntersection();
    }
}

rtDeclareVariable(float3, skyColor, ,);

RT_PROGRAM void miss() {
    rayData.result += rayData.attenuation * skyColor;
    rayData.stop = true;
}

rtDeclareVariable(ShadowPayload, shadowPayload, rtPayload, );

RT_PROGRAM void shadow() {
    if ((mat.visible && !hideLights) ||  (mat.visible && hideLights && fmaxf(mat.emission) < 0.01f)) shadowPayload.hit = true;
}