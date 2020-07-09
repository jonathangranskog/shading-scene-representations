#pragma once

#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <optix_world.h>
#include <cassert>
#include <string>

using namespace optix;

float4 readFloat4(const rapidjson::Value& value) {
    assert(value.IsArray() && value.Size() == 4);
    float x = value[0].GetFloat();
    float y = value[1].GetFloat();
    float z = value[2].GetFloat(); 
    float w = value[3].GetFloat();
    return make_float4(x, y, z, w);
}

float3 readFloat3(const rapidjson::Value& value) {
    assert(value.IsArray() && value.Size() == 3);
    float x = value[0].GetFloat();
    float y = value[1].GetFloat();
    float z = value[2].GetFloat();
    return make_float3(x, y, z);
}

float2 readFloat2(const rapidjson::Value& value) {
    assert(value.IsArray() && value.Size() == 2);
    float x = value[0].GetFloat();
    float y = value[1].GetFloat();
    return make_float2(x, y);
}

float readFloat(const rapidjson::Value& value) {
    assert(value.IsFloat());
    float v = value.GetFloat();
    return v;
}

int readInt(const rapidjson::Value& value) { 
    assert(value.IsInt());
    int v = value.GetInt();
    return v;
}

bool readBoolean(const rapidjson::Value& value) {
    assert(value.IsBool());
    bool v = value.GetBool();
    return v;
}

std::string readString(const rapidjson::Value& value) {
    assert(value.IsString());
    std::string s = value.GetString();
    return s;
}

Matrix4x4 readMat4(const rapidjson::Value& value) {
    assert(value.IsArray() && value.Size() == 4);
    Matrix4x4 m = Matrix4x4::identity();

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            m[j + i * 4] = value[j][i].GetFloat();
        }
    }

    return m;
}