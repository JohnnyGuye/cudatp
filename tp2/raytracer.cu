#include "wb.h"

#define SQR(x) ((x)*(x))

#include "cuda.def.cuh"
#include "vector3.h"
#include <fstream>
#include <cmath>




class Sphere {
public:
	float size = 1;
	Vector3 center;
        Vector3 color;

        GCPU_F Sphere()
            : size(1), color( 1.0, 0.3, 0.4 ), center( 0.0 ){}

        GCPU_F Sphere( Sphere & sphere )
            : size( sphere.size ), center( sphere.center ), color( sphere.color ) {}

};


__global__ void raytrace( Vector3 * image, Sphere * spheres, int width, int height, int sphereCount ) {

        int idx = threadIdx.x
                + threadIdx.y * blockDim.x
                + blockIdx.x * blockDim.x * blockDim.y;

        printf("%d\n", threadIdx.x + threadIdx.y * blockDim.x);
        if( width * height < idx ) return;

        int i = idx % width;
        int j = idx / height;
	
        printf("-- %d %d %d\n", spheres[0].center.r, spheres[0].center.g, spheres[0].center.b );
        Vector3 pos = Vector3( i, j, 0 );
        auto sphere = spheres[0];
        auto dist2 = Vector3::distance2( pos, sphere.center );
        auto sqrSize2 = SQR( sphere.size );

        auto a = (image[ j * width + i ] = Vector3( height,123,156 ) );
        printf("%d\n", a.r  );
        if( dist2 < sqrSize2 ){
		
                image[ j * width + i ] = Vector3( 1 - (j / (float)height) );

        } else {

//                //image[ j * width + i ] = sphere.color;
//                //image[ j * width + i ].r *= 1 - dist2 / sqrSize2;
//                //image[ j * width + i ].g *= 1 - dist2 / sqrSize2;
//                //image[ j * width + i ].b *= 1 - dist2 / sqrSize2;
	
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

class Scene {
public:
    Sphere * spheres;
    int sphereCount = 0;

    GCPU_F Scene() {}

    GCPU_F ~Scene() {
        if( sphereCount != 0 ) delete [] spheres;
    }

    GCPU_F Scene & addSphere( const Sphere & sphere ) {
        Sphere * spheres = new Sphere[ sphereCount + 1 ];
        for( unsigned int i = 0; i < sphereCount; i++ ) {
            spheres[i] = this->spheres[i];
        }
        spheres[sphereCount] = sphere;
        if( sphereCount ) {
            delete [] spheres;
        }
        this->spheres = spheres;
        sphereCount++;
        return *this;
    }

};

void populateScene( Scene & scene ) {

    Sphere sphere;
    sphere.center = Vector3( 5, 5, 0 );
//    sphere.center.r = 50;
//    sphere.center.g = 50;
//    sphere.size = 20;

    scene.addSphere( sphere );

}

int main(int argc, char *argv[]) {

        cudaDeviceReset();

	Vector3 * imgData;

	Vector3 * d_imgData;
        Sphere * d_spheres;

        int width = 10;
        int height = 10;
        auto pixelCount = width * height;

        imgData = new Vector3[ width * height ];

        Scene scene;
        populateScene( scene );
	
        std::cout << scene.spheres[0].center << std::endl;

        cudaMalloc( (void **) &d_imgData, pixelCount * sizeof( Vector3 ) );
        cudaMalloc( (void **) &d_spheres, scene.sphereCount * sizeof( Sphere ) );

        cudaMemcpy( d_spheres, scene.spheres, scene.sphereCount * sizeof( Sphere ), cudaMemcpyHostToDevice );

        dim3 threaddim = dim3( 32, 32, 1 );
        auto totalBlockCount = ceil( pixelCount / (float)(threaddim.x * threaddim.y) );
        dim3 blockdim = dim3( totalBlockCount, 1, 1 );

        raytrace<<< blockdim, threaddim >>>( d_imgData, scene.spheres, width, height, scene.sphereCount );

        CHECK_ERROR

        cudaDeviceSynchronize ();

        cudaMemcpy( imgData, d_imgData, width * height * sizeof( Vector3 ), cudaMemcpyDeviceToHost );

        CHECK_ERROR

        cudaFree( d_imgData );
        cudaFree( d_spheres );

        printf("Yolo");


        writeP3( "output.ppm", imgData, width, height );

        delete [] imgData;
}
