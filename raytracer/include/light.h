#include "cuda.def.cuh"
#include "vector3.h"

class Light {
public:
    Vector3 origin;
    Vector3 color = 1;
    float intensity = 1;

    GCPU_F Light() {}
};