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

    GCPU_F static Vector3 reflect( const Vector3 & I, const Vector3 & N ) {
        return I - N * (2.0 * N.dot(I));
    }

    GCPU_F static Vector3 refract( const Vector3 & I, const Vector3 & N, const float eta ) {
        auto NdotI = N.dot(I);
        auto k = 1.0 - SQR(eta) * (1.0 - SQR(NdotI));
        if( k < 0.0 ) return Vector3();
        return I * eta - N * (NdotI * eta + sqrt(k));
    }
};
