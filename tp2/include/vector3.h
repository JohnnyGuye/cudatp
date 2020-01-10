#pragma once

#include <iostream>

#define LOOP_I for(unsigned i = 0; i < 3; i++)
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
            //printf("Triple call %d\n", r);
        }

        GCPU_F Vector3( float rgb = 0 )
        : r(rgb), g(rgb), b(rgb) {
            //printf("Single call %d\n", r);
        }

        GCPU_F Vector3( const Vector3 & v )
        : r(v.r), g(v.g), b(v.b){
            //printf("Copy call %d\n", r);
        }

        GCPU_F Vector3 & operator=( const Vector3 & v ){
            this->r = v.r;
            this->g = v.g;
            this->b = v.b;
            //printf("Assign call %d\n", r);
            return *this;
        }

        GCPU_F float & operator[]( const int i ) {
            return values[i];
        }

        GCPU_F float operator[]( const int i ) const {
            return values[i];
        }

        GCPU_F Vector3 & operator *= ( const float s ) {
            LOOP_I values[i] *= s;
            return *this;
        }

        GCPU_F Vector3 & operator /= ( const float s ) {
            LOOP_I values[i] /= s;
            return *this;
        }

        GCPU_F Vector3 & operator += ( const float s ) {
            LOOP_I values[i] += s;
            return *this;
        }

        GCPU_F Vector3 & operator -= ( const float s ) {
            LOOP_I values[i] -= s;
            return *this;
        }

        GCPU_F Vector3 & operator += ( const Vector3 & v ) {
            LOOP_I values[i] += v[i];
            return *this;
        }

        GCPU_F Vector3 & operator -= ( const Vector3 & v ) {
            LOOP_I values[i] -= v[i];
            return *this;
        }

        GCPU_F Vector3 & operator *= ( const Vector3 & v ) {
            LOOP_I values[i] *= v[i];
            return *this;
        }

        GCPU_F Vector3 & operator /= ( const Vector3 & v ) {
            LOOP_I values[i] /= v[i];
            return *this;
        }

        GCPU_F Vector3 & operator - () {
            LOOP_I values[i] = -values[i];
            return *this;
        }

        GCPU_F Vector3 operator + ( const float s ) const {
            auto v = *this;
            v += s;
            return v;
        }

        GCPU_F Vector3 operator - ( const float s ) const {
            auto v = *this;
            v -= s;
            return v;
        }
        GCPU_F Vector3 operator * ( const float s ) const {
            auto v = *this;
            v *= s;
            return v;
        }
        GCPU_F Vector3 operator / ( const float s ) const {
            auto v = *this;
            v /= s;
            return v;
        }

        GCPU_F Vector3 operator + ( const Vector3 & oth ) const {
            auto v = *this;
            v += oth;
            return v;
        }

        GCPU_F Vector3 operator - ( const Vector3 & oth ) const {
            auto v = *this;
            v -= oth;
            return v;
        }

        GCPU_F Vector3 operator * ( const Vector3 & oth ) const {
            auto v = *this;
            v *= oth;
            return v;
        }

        GCPU_F Vector3 operator / ( const Vector3 & oth ) const {
            auto v = *this;
            v /= oth;
            return v;
        }

        GCPU_F float length2() const {
            float n = 0;
            LOOP_I n += SQR( values[i] );
            return n;
        }

        GCPU_F float length() const {
            return sqrt( length2() );
        }

        GCPU_F Vector3 & normalize() {
            auto l = length();
            if( l ) {
                *this /= l;
            }
            return *this;
        }

        GCPU_F Vector3 normalized() const {
            auto v = *this;
            return v.normalize();
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

        GCPU_F Vector3 & clamp( const float min, const float max ) {
            LOOP_I {
                if( values[i] <= min ) values[i] = min;
                if( values[i] >= max ) values[i] = max;
            }

            return *this;
        }

        friend std::ostream & operator << ( std::ostream & os, const Vector3 & vec ) {
            os << "[" << vec.r << "; " << vec.g << "; " << vec.b << "]";
            return os;
        }


};
