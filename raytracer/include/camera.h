#include "cuda.def.cuh"

class Camera {
public:
    Vector3 resolution  = Vector3(100,100,1);
    Vector3 direction   = Vector3(0,0,1);
    Vector3 up          = Vector3(0,1,0);
    Vector3 position    = Vector3(0,0,-100);

    bool orthogonal = true;
    Vector3 size = Vector3(100,100,0);

    GCPU_F Ray castRayFromPixel( const int x, const int y ) const {
        if( orthogonal ) {
            Ray ray;
            ray.origin = (Vector3(x,y,0) / resolution - 0.5f) * size + position;
            ray.direction = direction;
            return ray;
        } else {
            Ray ray;
            return ray;
        }
    }

};
