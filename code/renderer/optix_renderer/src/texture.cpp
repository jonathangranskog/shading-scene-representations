#include "texture.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include <iostream>

using namespace optix;

// Load an image into an optix sampler
TextureSampler loadTexture(const std::string& path, Context context) {
    // Load image
    int c, h, w;
    float* out = stbi_loadf(path.c_str(), &w, &h, &c, 0);

    // Prepare texture sampler
    TextureSampler sampler = context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_REPEAT);
    sampler->setWrapMode(1, RT_WRAP_REPEAT);

    // Copy data from float array
    std::vector<float4> data = std::vector<float4>(w * h);
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            float4 color = make_float4(0, 0, 0, 1);
            if (c > 0) color.x = out[j * w * c + i * c + 0];
            if (c > 1) color.y = out[j * w * c + i * c + 1];
            if (c > 2) color.z = out[j * w * c + i * c + 2];
            if (c > 3) color.w = out[j * w * c + i * c + 3];
            data[j * w + i] = color;
        }
    }

    // Create buffer
    Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, w, h);
    memcpy(buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), data.data(), w * h * sizeof(float4));
    buffer->unmap();

    // Set sampler buffer
    sampler->setBuffer(buffer);
    delete[] out;
    return sampler;
}