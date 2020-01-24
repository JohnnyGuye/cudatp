#include "cuda.def.cuh"

#include "vector3.h"
#include "material.h"

class Sphere {
public:
        float size = 1;
        Vector3 center;
        Material material;

        GCPU_F Sphere()
            : size(1), center( 0.0 ){}

        GCPU_F Sphere( Sphere & sphere )
            : size( sphere.size ), center( sphere.center ), material( sphere.material ) {}

        GCPU_F Vector3 normalAt( const Vector3 & hit ) const {
            return (hit - center).normalize();
        }

        GCPU_F bool intersects( const Vector3 & origin, const Vector3 & direction, Vector3 & hit, float & distance ) const {
            auto m = origin - center;
            float b = Vector3::dot( m, direction );
            float c = m.length2() - SQR( size );

            if( c > 0.0f && b > 0.0f ) return false;
            float discr = SQR( b ) - c;

            if( discr < 0.0f ) return false;
            auto t = -b - sqrt( discr );

            if( t < 0.0f ) t = -b + sqrt( discr );

            hit = origin + direction * t;
            distance = t;

            return true;
        }

        friend std::ostream & operator << ( std::ostream & os, const Sphere & sphere ) {
            os << "Sphere: radius: " << sphere.size
               << " center: " << sphere.center
               << " color: " << sphere.material.albedo;
            return os;
        }
};
