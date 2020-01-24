#include "cuda.def.cuh"

class Ray {
public:
    Vector3 origin;
    Vector3 direction;

    GCPU_F Ray( const Vector3 & origin, const Vector3 & direction )
        : origin( origin ), direction( direction ) {}

    GCPU_F Ray( const Vector3 & direction )
        : Ray( 0, direction ) {}

    GCPU_F Ray( )
        : Ray( Vector3(1, 0, 0) ) {}

};