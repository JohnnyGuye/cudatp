#pragma once

#include <iostream>

#define LOOP_I for(unsigned i = 0; i < 3; i++)

struct Vector3 {

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
            struct {
                float u;
                float v;
                float s;
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

        GCPU_F bool operator==( const Vector3 & v ) {
            LOOP_I if( v[i] != values[i]) return false;
            return true;
        }

        GCPU_F bool operator!=( const Vector3 & v ) {
            return !(*this == v);
        }

        GCPU_F Vector3 & operator=( const Vector3 & v ){
            this->r = v.r;
            this->g = v.g;
            this->b = v.b;
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

        GCPU_F float dot( const Vector3 & v ) const {
            float s = 0;
            LOOP_I s += v.values[i] * values[i];
            return s;
        }

        GCPU_F float length2() const {
            return dot(*this);
        }

        GCPU_F float length() const {
            return sqrt( length2() );
        }

        GCPU_F Vector3 & abs() {
            LOOP_I values[i] = values[i] >= 0 ? values[i] : -values[i];
            return *this;
        }

        GCPU_F Vector3 absed() const {
            auto v = *this;
            return v.abs();
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

        GCPU_F static float dot( const Vector3 & lhs, const Vector3 & rhs ) {
                return lhs.dot(rhs);
        }

        GCPU_F  static Vector3 normalized( const Vector3 & v ) {
                auto sqr = v.length();
                return Vector3( v.r / sqr, v.g / sqr, v.b / sqr );
        }

        GCPU_F Vector3 & clamp( const float min, const float max ) {
            LOOP_I {
                if( values[i] <= min ) values[i] = min;
                if( values[i] >= max ) values[i] = max;
            }

            return *this;
        }

        GCPU_F Vector3 & clamp( const Vector3 & min, const Vector3 & max ) {
            LOOP_I {
                if( values[i] <= min[i] ) values[i] = min[i];
                if( values[i] >= max[i] ) values[i] = max[i];
            }

            return *this;
        }

        GCPU_F static Vector3 clamp( Vector3 val, const Vector3 & min, const Vector3 & max ) {
            return val.clamp(min,max);
        }

        GCPU_F static Vector3 mix( const Vector3 & a, const Vector3 & b, const float ratio ) {
            return a * (1-ratio) + b * ratio;
        }


        friend std::ostream & operator << ( std::ostream & os, const Vector3 & vec ) {
            os << "[" << vec.r << "; " << vec.g << "; " << vec.b << "]";
            return os;
        }


};


#undef LOOP_I
