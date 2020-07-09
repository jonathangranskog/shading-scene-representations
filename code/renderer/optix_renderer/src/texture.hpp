#pragma once

#include <optix_world.h>
#include <vector>

using namespace optix;

TextureSampler loadTexture(const std::string& path, Context context);