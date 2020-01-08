#pragma once

#include <iostream>

class Vector3 {
public:
        union {
            struct {
                float values[3];
            };
            struct {
                float r;
                float g;
                float b;
            };
            struct {
                float x;
                float y;
                float z;
            };
        };

        GCPU_F Vector3( float r, float g, float b )
        : r(r), g(g), b(b) {
            printf("Triple call %d\n", r);
        }

        GCPU_F Vector3( float rgb = 0 )
        : r(rgb), g(rgb), b(rgb) {
            printf("Single call %d\n", r);
        }

        GCPU_F Vector3( const Vector3 & v )
        : r(v.r), g(v.g), b(v.b){
            printf("Copy call %d\n", r);
        }

        GCPU_F Vector3 & operator=( const Vector3 & v ){
            this->r = v.r;
            this->g = v.g;
            this->b = v.b;
            printf("Assign call %d\n", r);
            return *this;
        }

        GCPU_F static float distance2( const Vector3 & lhs, const Vector3 &rhs ) {
                return SQR(lhs.r - rhs.r) + SQR(lhs.g - rhs.g) + SQR(lhs.b - rhs.b);
        }

        GCPU_F static float dotProduct( const Vector3 & lhs, const Vector3 & rhs ) {
                return lhs.r * rhs.r + lhs.g * rhs.g + lhs.b * rhs.b;
        }

        GCPU_F  static Vector3 normalized( const Vector3 & v ) {
                auto sqr = sqrt( Vector3::dotProduct( v, v ) );
                return Vector3( v.r / sqr, v.g / sqr, v.b / sqr );
        }

        GCPU_F float operator[]( int i ) {
            return values[i];
        }

        friend std::ostream & operator << ( std::ostream & os, const Vector3 & vec ) {
            os << "[" << vec.r << "; " << vec.g << "; " << vec.b << "]";
            return os;
        }


};
