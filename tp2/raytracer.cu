#include "wb.h"

#define SQR(x) ((x)*(x))

#include "cuda.def.cuh"
#include "vector3.h"
#include <fstream>
#include <cmath>




class Sphere {
public:
	float size;
	Vector3 center;
        Vector3 color;

        GCPU_F Sphere()
            : size(1), color( 1.0, 0.3, 0.4 ), center( 0.0 ){}

        GCPU_F Sphere( Sphere & sphere )
            : size( sphere.size ), center( sphere.center ), color( sphere.color ) {}

};


__global__ void raytrace( Vector3 * image, Sphere * spheres, int width, int height, int sphereCount ) {

        int idx = threadIdx.x
                + threadIdx.y * blockDim.y
                + blockIdx.x * blockDim.x * blockDim.y;

	if( width * height <= idx ) return;
        
	//printf("%d\n", threadIdx.x + threadIdx.y * blockDim.x);
        

        int i = idx % width;
        int j = idx / height;

	//printf("%d %d %d\n", i, j, idx ); 
	
        //printf("-- %d %d %d\n", spheres[0].center.r, spheres[0].center.g, spheres[0].center.b );
        Vector3 pos = Vector3( i, j, 0 );

        image[ idx ] = Vector3( 1 - (j / (float)height) );

        for( auto i = 0; i < sphereCount; i++ ) {

            auto sphere = spheres[i];
            auto dist2 = Vector3::distance2( pos, sphere.center );
            auto sqrSize2 = SQR( sphere.size );

            //auto a = (image[ idx ] = Vector3( height,123,156 ) );
            //printf("%d\n", a.r  );
            if( dist2 < sqrSize2 ){

                    image[ idx ] = sphere.color;
                    image[ idx ].r *= 1 - dist2 / sqrSize2;
                    image[ idx ].g *= 1 - dist2 / sqrSize2;
                    image[ idx ].b *= 1 - dist2 / sqrSize2;

            }

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

    Sphere sphere1;
    sphere1.center = Vector3( 140, 130, 0 );
    sphere1.size = 100;

    Sphere sphere2;
    sphere2.center = Vector3( 210, 230, 0 );
    sphere2.size = 20;

    scene.addSphere( sphere2 );
    scene.addSphere( sphere1 );

}

int main(int argc, char *argv[]) {

        cudaDeviceReset();

	Vector3 * imgData;

	Vector3 * d_imgData;
        Sphere * d_spheres;

        int width = 300;
        int height = 300;
        auto pixelCount = width * height;

        imgData = new Vector3[ pixelCount ];

        Scene scene;
        populateScene( scene );
	
        cudaMalloc( (void **) &d_imgData, pixelCount * sizeof( Vector3 ) );
        cudaMalloc( (void **) &d_spheres, scene.sphereCount * sizeof( Sphere ) );

        cudaMemcpy( d_spheres, scene.spheres, scene.sphereCount * sizeof( Sphere ), cudaMemcpyHostToDevice );

        dim3 threaddim = dim3( 32, 32, 1 );
        auto totalBlockCount = ceil( pixelCount / (float)(threaddim.x * threaddim.y) );
        dim3 blockdim = dim3( totalBlockCount, 1, 1 );

        raytrace<<< blockdim, threaddim >>>( d_imgData, d_spheres, width, height, scene.sphereCount );

        CHECK_ERROR

        cudaDeviceSynchronize ();

        cudaMemcpy( imgData, d_imgData, pixelCount * sizeof( Vector3 ), cudaMemcpyDeviceToHost );

        CHECK_ERROR

        cudaFree( d_imgData );
        cudaFree( d_spheres );

        writeP3( "output.ppm", imgData, width, height );

        delete [] imgData;
}
