#include "wb.h"

#include <fstream>
#include <cmath>

#define SQR(x) ((x)*(x))

#define GCPU_F __host__ __device__

class Vector3 {
public:
        union {
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
                float values[3];
            };
        };

        GCPU_F Vector3( float r, float g, float b )
	: r(r), g(g), b(b) {}

        GCPU_F Vector3( float rgb = 0 )
	: r(rgb), g(rgb), b(rgb) {}
	
        GCPU_F Vector3( const Vector3 & v )
        : Vector3(v.r, v.g, v.b ){}

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


};

class Sphere {
public:
	float size = 1;
	Vector3 center;
	Vector3 color = Vector3( 1.0, 0.3, 0.4 );

        GCPU_F Sphere() {}

        GCPU_F Sphere( Sphere & sphere )
            : size( sphere.size ), center( sphere.center ), color( sphere.color ) {}

};


__global__ void raytrace( Vector3 * image, Sphere * spheres, int width, int height, int sphereCount ) {

	int i = threadIdx.x;
	int j = blockIdx.x;

	if( i > width ) return;
	if( j > height ) return;
	
	Vector3 pos = Vector3( i, j, 0 );
        auto sphere = spheres[0];
        auto dist2 = Vector3::distance2( pos, sphere.center );
	auto sqrSize2 = SQR( sphere.size );
        if( dist2 < sqrSize2 ){
		
		image[ j * width + i ] = Vector3( 1 - (j / (float)height) );		

	} else {

		image[ j * width + i ] = sphere.color;
		image[ j * width + i ].r *= 1 - dist2 / sqrSize2;
		image[ j * width + i ].g *= 1 - dist2 / sqrSize2;
		image[ j * width + i ].b *= 1 - dist2 / sqrSize2;
	
	}
	
}

void writeP3( std::string fileName, Vector3 * data, int width, int height ) {
	
	std::ofstream file;
	file.open( fileName );
	file << "P3\n" << width << " " << height << "\n" << 255 << std::endl;
	
	for( auto idx = 0; idx < width * height; idx++ ) {
		file << (int)(data[ idx ].r * 255) << " "
			<< (int)(data[ idx ].g * 255) << " "
			<< (int)(data[ idx ].b * 255) << " ";
	}

	file << std::flush;

}

int main(int argc, char *argv[]) {

	Vector3 * imgData;
	Vector3 * d_imgData;

        int width = 100;
        int height = 100;

        Sphere * spheres = new Sphere[1];
        Sphere * d_spheres;

	Sphere sphere;
        sphere.center.r = 50;
        sphere.center.g = 50;
        sphere.size		= 20;
	
        spheres[0] = sphere;

	imgData = new Vector3[ width * height ];
	
	cudaMalloc( (void **) &d_imgData, width * height * sizeof( Vector3 ) );	
        cudaMalloc( (void **) &d_spheres, 1 * sizeof( Sphere ) );

        cudaMemcpy( d_spheres, spheres, 1 * sizeof( Sphere ), cudaMemcpyHostToDevice );

        raytrace<<< 100, 100 >>>( d_imgData, spheres, width, height, 1 );

	cudaMemcpy( imgData, d_imgData, width * height * sizeof( Vector3 ), cudaMemcpyDeviceToHost );	

	writeP3( "output.ppm", imgData, width, height );

	delete [] imgData;
        delete [] spheres;
}
