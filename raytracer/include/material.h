#pragma once

#include "cuda.def.cuh"

class Material {
public:
    Vector3 albedo;
    float roughness = 0;
    float metalness = 0;
    float opacity   = 1;
};
