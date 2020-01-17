#include "wb.h"

#define SQR(x) ((x)*(x))
#define SQR5(x) ((x)*(x)*(x)*(x)*(x))

#include "cuda.def.cuh"
#include "vector3.h"
#include <fstream>
#include <cmath>


// ==========================================================
// ============ COMPILATION OPTIONS =========================
// ==========================================================

const bool OPT_CPU_MAXLUMINANCE = false;

// ==========================================================
// ============ CLASSES DEFINITIONS =========================
// ==========================================================

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

class Material {
public:
    Vector3 albedo;
    float roughness = 0;
    float metalness = 0;
};

class Camera {
public:
    Vector3 resolution  = Vector3(100,100,1);
    Vector3 direction   = Vector3(0,0,1);
    Vector3 up          = Vector3(0,1,0);
    Vector3 position    = Vector3(0,0,-100);

    bool orthogonal = true;
    Vector3 size = Vector3(100,100,0);

    GCPU_F Ray castRayFromPixel( int x, int y ) {
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

class Light {
public:
    Vector3 origin;
    Vector3 color = 1;
    float intensity = 1;

    Light() {}
};

class Collision {
public:

    Sphere * sphere = nullptr;
    Vector3 normal;
    Vector3 hit;
    float distance = 2 * 1e38;

};

GCPU_F bool getFirstCollision( const Ray & ray, Sphere * spheres, const int sphereCount, Collision & collision ) {

    collision.distance = 2e38;
    collision.sphere = nullptr;

    auto collided = false;

    for( unsigned int i = 0; i < sphereCount; i++ ) {
        Vector3 hit;
        float distance;
        if( !spheres[i].intersects( ray.origin, ray.direction, hit, distance ) ) continue;
        if( collision.distance < distance ) continue;

        collision.distance = distance;
        collision.hit = hit;
        collision.normal = spheres[i].normalAt( hit );
        collision.sphere = &spheres[i];
        collided = true;

    }

    return collided;
}

GCPU_F float GGXDistribution( float alpha2, float NdotH ) {
    auto denom = SQR( NdotH ) * (alpha2 - 1) + 1;
    return alpha2 / (M_PI * SQR( denom ));
}

GCPU_F float GGXGeometry( float alpha2, float NdotL ) {
    return 2 * NdotL / (NdotL + sqrt(alpha2 + (1-alpha2) * SQR(NdotL) ));
}

GCPU_F Vector3 Fresnel( const Vector3 & f0, const float NdotH ) {
    auto unary = Vector3(1);
    return f0 + (unary - f0) * (SQR5(unary - NdotH));
}

GCPU_F float dodgeZero( float value ) {
    if( value  < 0 ) value = 0;
    if( value < 0.001 ) value = 0.001;
    return value;
}

GCPU_F float computeLuminance( const Vector3 & color ) {
    return color.dot(Vector3(0.2126f, 0.7152f, 0.0722f));
}

__global__ void raytrace( Vector3 * image, Sphere * spheres, Light * lights, Camera camera, int sphereCount, int lightCount ) {

        int width = camera.resolution.x;
        int height = camera.resolution.y;

        int idx = threadIdx.x
                + threadIdx.y * blockDim.y
                + blockIdx.x * blockDim.x * blockDim.y;

        if( idx >= width * height ) return;
        
	//printf("%d\n", threadIdx.x + threadIdx.y * blockDim.x);
        

        int i = idx % width;
        int j = height - idx / height;

//        printf("%d %d %d\n", i, j, idx );
	
        //printf("-- %d %d %d\n", spheres[0].center.r, spheres[0].center.g, spheres[0].center.b );

        Ray ray;
        ray.origin = Vector3( i, j, -100 );
        ray.direction = Vector3( 0, 0, 1 );

        ray = camera.castRayFromPixel( i, j );

//        printf("%f %f %f\n", ray.origin.x, ray.origin.y, ray.origin.z);
//        image[ idx ] = (ray.origin / camera.resolution * 2 + 1).abs();
        image[ idx ] = ((ray.origin - camera.position) / camera.size).abs();

        Collision collision;
        if( !getFirstCollision( ray, spheres, sphereCount, collision ) ) return;

        //image[ idx ] = collision.sphere->color;

        auto material = collision.sphere->material;
        auto albedo = material.albedo;
        auto rough = material.roughness * 0.97 + 0.03;
        auto metal = material.metalness;
        auto f0 = albedo * (0.04 * (1-metal) + metal);

        auto alpha = SQR( rough );
        auto alpha2 = SQR( alpha );

        image[ idx ] = 0;

        for( unsigned int i = 0; i < lightCount; i++ ) {
            auto & light = lights[i];
            auto color = light.color * light.intensity;
            auto L = (light.origin - collision.hit).normalize();
            auto N = collision.normal;
            auto V = -ray.direction.normalized();
            auto H = (L + V).normalize();

            auto NdotL = Vector3::dot( N, L );

            if( NdotL <= 0 ) continue;

            auto NdotV = dodgeZero( N.dot(V) );
            auto NdotH = dodgeZero( N.dot(H) );

            NdotL = dodgeZero( NdotL );

            auto D = GGXDistribution( alpha2, NdotL );
            auto G = GGXGeometry( alpha2, NdotL ) * GGXGeometry( alpha2, NdotV );
            auto F = Fresnel( f0, NdotH );
            auto denom = 4 * NdotL * NdotV;

            auto spec = F * G * D / denom;
            auto nonspec = (Vector3(1) - F);

            color *= albedo * (spec + nonspec) / M_PI * NdotL;

            image[idx] += color;
        }	
}

__global__ void tonemap( Vector3 * image, float maxLuminance, Camera camera ) {

    int width = camera.resolution.x;
    int height = camera.resolution.y;

    int idx = threadIdx.x
            + threadIdx.y * blockDim.y
            + blockIdx.x * blockDim.x * blockDim.y;

    if( idx >= width * height ) return;

//    int i = idx % width;
//    int j = height - idx / height;

    auto coeff = (1 + maxLuminance) / maxLuminance;
    auto luminance = computeLuminance(image[idx]);
    image[idx] = image[idx] * luminance / (1 + luminance) * coeff;

}

// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template< int VERS >
__global__ void computeMaxLuminance( const Vector3 * in_colors, Vector3 * out_colors, const Camera camera ) {

    extern __shared__ Vector3 sharedData[];

    unsigned int tid    = threadIdx.x;
    unsigned int i      = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int px     = camera.resolution.x * camera.resolution.y;
    if( i >= px || tid >= px ) return;

    sharedData[ tid ] = in_colors[ i ];
    __syncthreads();

    auto lumiL = computeLuminance( sharedData[tid] );

    // The reduction
    for( unsigned int s = 1; s < blockDim.x; s *= 2 ) {
        if( tid % (2*s) == 0) {
            auto lumiR = computeLuminance( sharedData[tid+s] );
            if( lumiR > lumiL )
                sharedData[ tid ] = sharedData[tid+s];
        }
        __syncthreads();
    }

    if( tid == 0 ) out_colors[blockIdx.x] = sharedData[0];
}

void writeP3( std::string fileName, Vector3 * data, int width, int height ) {
	
	std::ofstream file;
	file.open( fileName );
	file << "P3\n" << width << " " << height << "\n" << 255 << std::endl;
	
        for( auto idx = 0; idx < width * height; idx++ ) {
            auto d = data[ idx ];
            d.clamp( 0, 1 );
//            std::cout << d * 255 << std::endl;
            file << (int)(d.r * 255) << " "
                    << (int)(d.g * 255) << " "
                    << (int)(d.b * 255) << " ";
	}

	file << std::flush;

}

class Scene {
public:

    Light * lights;
    int lightCount = 0;

    Sphere * spheres;
    int sphereCount = 0;

    GCPU_F Scene() {}

    GCPU_F ~Scene() {
        if( sphereCount != 0 ) delete [] spheres;
        if( lightCount != 0 ) delete [] lights;
    }

    Scene & addSphere( const Sphere & sphere ) {
        Sphere * spheres = new Sphere[ sphereCount + 1 ];
        for( unsigned int i = 0; i < sphereCount; i++ ) {
            spheres[i] = this->spheres[i];
        }

        spheres[sphereCount] = sphere;
        if( sphereCount ) {
            delete [] this->spheres;
        }

        this->spheres = spheres;
        sphereCount++;
        return *this;
    }

    Scene & addLight( const Light & light ) {
        Light * lights = new Light[ lightCount + 1 ];
        for( unsigned int i = 0; i < lightCount; i++ ) {
            lights[i] = this->lights[i];
        }

        lights[lightCount] = light;
        if( lightCount ) {
            delete [] this->lights;
        }

        this->lights = lights;
        lightCount++;
        return *this;
    }

};

void populateScene( Scene & scene ) {

    auto steps = 5;
    auto stepW = 300 / (steps + 2);
    auto stepH = 300 / (steps + 2);
    for( unsigned int i = 0; i <= steps; i++ ) {
        for( unsigned int j = 0; j <= steps; j++ ) {

            Sphere sphere;
            sphere.center   = Vector3( stepW * (i + 1), stepH * (j + 1), 0 );
            sphere.size     = min( stepW, stepH ) / 2.5f;

            sphere.material.albedo = Vector3(1, 0.6, 0.8);
            sphere.material.roughness = (float)i/(float)steps;
            sphere.material.metalness = (float)j/(float)steps;

            scene.addSphere( sphere );
        }
    }
//    Sphere sphere1;
//    sphere1.center = Vector3( 140, 130, 0 );
//    sphere1.size = 100;
//    sphere1.material.albedo = Vector3( 1, 0, 0 );

//    Sphere sphere2;
//    sphere2.center = Vector3( 210, 230, 0 );
//    sphere2.size = 20;
//    sphere2.material.albedo = Vector3( 0, 0, 1 );

//    scene.addSphere( sphere1 );

//    scene.addSphere( sphere2 );

    Light light1;
    light1.origin = Vector3( 0, 10000, -10000 );
    light1.color = Vector3( 0.6, 0.8, 1 );
    light1.intensity = 5;

    Light light2;
    light2.origin = Vector3( 500, 500, -500 );
    light2.color = Vector3( 1, 0.8, 0.6 );
    light2.intensity = 5;

    scene.addLight( light1 );
    scene.addLight( light2 );

}


int main(int argc, char *argv[]) {

        cudaDeviceReset();

	Vector3 * imgData;

	Vector3 * d_imgData;
        Sphere * d_spheres;
        Light * d_lights;

        Camera camera;
        Scene scene;
        populateScene( scene );

        camera.resolution = Vector3( 1024, 1024, 0) / 16  + Vector3(0,0,1);
        camera.size = Vector3( 300, 300, 1 );
        camera.position = Vector3( 150, 150, -100 );

        int width = camera.resolution.x;
        int height = camera.resolution.y;
        auto pixelCount = width * height;

        imgData = new Vector3[ pixelCount ];
	
        cudaMalloc( (void **) &d_imgData, pixelCount * sizeof( Vector3 ) );
        cudaMalloc( (void **) &d_spheres, scene.sphereCount * sizeof( Sphere ) );
        cudaMalloc( (void **) &d_lights, scene.lightCount * sizeof( Light ) );

        cudaMemcpy( d_spheres, scene.spheres, scene.sphereCount * sizeof( Sphere ), cudaMemcpyHostToDevice );
        cudaMemcpy( d_lights, scene.lights, scene.lightCount * sizeof( Light ), cudaMemcpyHostToDevice );

        dim3 threaddim = dim3( 32, 32, 1 );
        auto totalBlockCount = ceil( pixelCount / (float)(threaddim.x * threaddim.y) );
        dim3 blockdim = dim3( totalBlockCount, 1, 1 );

        raytrace<<< blockdim, threaddim >>>( d_imgData, d_spheres, d_lights, camera, scene.sphereCount, scene.lightCount );

        float maxLuminance = 0;

        if( OPT_CPU_MAXLUMINANCE ) {
            // CPU max
            cudaDeviceSynchronize();
            cudaMemcpy( imgData, d_imgData, pixelCount * sizeof( Vector3 ), cudaMemcpyDeviceToHost );
            for( unsigned int i = 0; i < pixelCount; i++ ) {
                auto luminance = computeLuminance( imgData[i] );
                if( luminance > maxLuminance ) maxLuminance = luminance;
            }
            cudaMemcpy( d_imgData, imgData, pixelCount * sizeof( Vector3 ), cudaMemcpyHostToDevice );

        } else {
            // GPU max
            dim3 threaddim = dim3( 256, 1, 1 );
            dim3 blockdim = dim3( ceil( pixelCount / (float)(threaddim.x) ), 1, 1 );

            Vector3 * d_reduction;
            Vector3 * d_reduction2;
            Vector3 * reduction = new Vector3[ pixelCount ];

            cudaDeviceSynchronize();
            cudaMalloc( (void **) &d_reduction, pixelCount * sizeof( Vector3 ) );
//            cudaMalloc( (void **) &d_reduction2, pixelCount * sizeof( Vector3 ) );
            CHECK_ERROR

            computeMaxLuminance<1><<< blockdim, threaddim, pixelCount * sizeof( Vector3 ) >>>( d_imgData, d_reduction, camera );

            CHECK_ERROR
            cudaMemcpy( imgData, d_imgData, pixelCount * sizeof( Vector3 ), cudaMemcpyDeviceToHost );
            CHECK_ERROR
            cudaMemcpy( reduction, d_reduction, pixelCount * sizeof( Vector3 ), cudaMemcpyDeviceToHost );
            CHECK_ERROR

            for( unsigned int i = 0; i < pixelCount; i += 2 ) {
                if( imgData[i] == reduction[i] ) continue;
                std::cout << "> " << i << std::endl;
                std::cout << imgData[i] << " " << imgData[i+1] << std::endl;
                std::cout << reduction[i] << " " << reduction[i+1] << std::endl;
            }
            cudaFree( d_reduction );
            CHECK_ERROR

            maxLuminance = 1;
        }

        CHECK_ERROR

        tonemap<<< blockdim, threaddim >>>( d_imgData, maxLuminance, camera );

        cudaDeviceSynchronize();
        cudaMemcpy( imgData, d_imgData, pixelCount * sizeof( Vector3 ), cudaMemcpyDeviceToHost );

        CHECK_ERROR

        cudaFree( d_imgData );
        cudaFree( d_spheres );
        cudaFree( d_lights );

        writeP3( "output.ppm", imgData, width, height );

        delete [] imgData;
}
