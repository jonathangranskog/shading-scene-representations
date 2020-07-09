#pragma once

#include <optix_world.h>
#include "math.h"

using namespace optix;

float3 hsv2rgb(const float3& hsv) {
    float hue = hsv.x * 360.f;
    float C = hsv.y * hsv.z;
    float X = C * (1 - fabs(fmod(hue / 60.f, 2.f) - 1));
    float m = hsv.z - C;

    float3 result;
    if (hue < 60) result = make_float3(C + m, X + m, m);
    else if (hue < 120) result = make_float3(X + m, C + m, m);
    else if (hue < 180) result = make_float3(m, C + m, X + m);
    else if (hue < 240) result = make_float3(m, X + m, C + m);
    else if (hue < 300) result = make_float3(X + m, m, C + m);
    else result = make_float3(C + m, m, X + m);

    return result;
}

Matrix4x4 createLookAt(float3 eye, float3 target, float3 up) {
    float3 forward = normalize(target - eye);
    float3 side = normalize(cross(forward, up));
    up = normalize(cross(side, forward));

    Matrix4x4 matrix;
    matrix.setCol(0, make_float4(side.x, up.x, -forward.x, 0.f));
    matrix.setCol(1, make_float4(side.y, up.y, -forward.y, 0.f));
    matrix.setCol(2, make_float4(side.z, up.z, -forward.z, 0.f));
    matrix.setCol(3, make_float4(-dot(side, eye), -dot(up, eye), dot(forward, eye), 1.f));
    return matrix;
}

Matrix4x4 createPerspective(float fovy, float aspect, float near, float far) {
    float top = near * tanf(fovy * M_PIf / 360.f);
    float right = top * aspect;
    float left = -right;
    float bottom = -top;

    float A = (right + left) / (right - left);
    float B = (top + bottom) / (top - bottom);
    float C = -(far + near) / (far - near);
    float D = -2.f * far * near / (far - near);
    float E = 2.f * near / (right - left);
    float F = 2.f * near / (top - bottom);

    optix::Matrix4x4 matrix;
    matrix.setCol(0, make_float4(E, 0, 0, 0));
    matrix.setCol(1, make_float4(0, F, 0, 0));
    matrix.setCol(2, make_float4(A, B, C, -1));
    matrix.setCol(3, make_float4(0, 0, D, 0));
    return matrix;
}