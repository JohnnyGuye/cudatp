#include "wb.h"

#define SQR(x) ((x)*(x))
#define SQR5(x) ((x)*(x)*(x)*(x)*(x))

#include "cuda.def.cuh"
#include "vector3.h"
#include "ray.h"
#include "camera.h"
#include "material.h"
#include "sphere.h"
#include "light.h"

#include <fstream>
#include <cmath>


// ==========================================================
// ============ COMPILATION OPTIONS =========================
// ==========================================================

constexpr bool OPT_CPU_MAXLUMINANCE = false;
constexpr int MAX_BOUNCE_DEPTH      = 4;

// ==========================================================
// ============ CLASSES DEFINITIONS =========================
// ==========================================================

constexpr float COLLISION_OFFSET = 1e-6;

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
    return f0 + (unary - f0) * (SQR5(1 - NdotH));
}

GCPU_F float dodgeZero( float value ) {
    if( value  < 0 ) value = 0;
    if( value < 0.001 ) value = 0.001;
    return value;
}

GCPU_F float computeLuminance( const Vector3 & color ) {
    return color.dot(Vector3(0.2126f, 0.7152f, 0.0722f));
}


/**
* Compute the light received on a specific point
* @param ray i
*/
__device__ Vector3 directLight( const Ray & ray, const Collision & collision, Sphere * spheres, Light * lights, const int sphereCount, const int lightCount) {

    Vector3 retColor;

    auto material = collision.sphere->material;
    auto albedo = material.albedo;
    auto rough = material.roughness * 0.97 + 0.03;
    auto metal = material.metalness;
    //auto f0 = albedo * (0.04 * (1-metal) + metal);
    auto f0 = Vector3::mix( Vector3(0.04), albedo, metal );

    auto alpha = SQR( rough );
    auto alpha2 = SQR( alpha );

    auto N = collision.normal;
    auto V = -ray.direction.normalized();
    auto NdotV = N.dot(V);
    if( NdotV < 0) {
        N = -N;
        NdotV = -NdotV;
    }
    NdotV = dodgeZero(NdotV);

    auto ggxNdotV = GGXGeometry( alpha2, NdotV );

    for( unsigned int i = 0; i < lightCount; i++ ) {
        auto & light = lights[i];
        auto radiance = light.color * light.intensity;
        auto L = (light.origin - collision.hit).normalize();
        auto H = (L + V).normalize();

        auto NdotL = Vector3::dot( N, L );

        if( NdotL <= 0 ) continue;

        auto NdotH = dodgeZero( N.dot(H) );

        NdotL = dodgeZero( NdotL );

        auto D = GGXDistribution( alpha2, NdotL );
        auto G = GGXGeometry( alpha2, NdotL ) * ggxNdotV;
        auto F = Fresnel( f0, NdotV );
        auto denom = 4 * NdotL * NdotV;

        auto spec = F * G * D / denom;

        auto kS = F;
        auto kD = (Vector3(1) - kS) * (1 - metal);

        auto color = (kD * albedo/ M_PI + spec) * radiance * NdotL;

        retColor += color;
    }

    return retColor;
}

/**
 * Follow a ray through the scene and gather its color bouncing back and forth on the objects
 *
 */
template< int DEPTH >
__device__ Vector3 trace(
            const Ray & ray,
            Sphere * spheres,
            Light * lights,
            const Camera camera,
            const int sphereCount,
            const int lightCount ) {

        Vector3 color;

        Collision collision;
        if( !getFirstCollision( ray, spheres, sphereCount, collision ) ) {
            if( DEPTH == 0 ) {
                color = ((ray.origin - camera.position) / camera.size).abs();
            } else {
                color = trace< MAX_BOUNCE_DEPTH >( ray, spheres, lights, camera, sphereCount, lightCount );
            }
            return color;
        }

        auto N = collision.normal;
        if( N.dot(ray.direction) > 0 ) {
            N = -N;
        }

        auto opacity = collision.sphere->material.opacity;

        Ray reflectedRay;
        reflectedRay.origin     = ray.origin + collision.normal * COLLISION_OFFSET;
        reflectedRay.direction  = Ray::reflect( ray.direction, collision.normal ).normalize();

        Ray refractedRay;
        refractedRay.origin     = ray.origin - collision.normal * COLLISION_OFFSET;
        refractedRay.direction  = Ray::refract( ray.direction, collision.normal, 1.0f ).normalize();

        auto reflLight = trace< DEPTH + 1 >( reflectedRay, spheres, lights, camera, sphereCount, lightCount ) * opacity;
        auto refrLight = trace< DEPTH + 1>( refractedRay, spheres, lights, camera, sphereCount, lightCount ) * (1 - opacity);
        auto iLight = reflLight + refrLight;
        auto dLight = directLight(ray, collision, spheres, lights, sphereCount, lightCount);

        return dLight + iLight;
}

template<>
__device__ Vector3 trace<MAX_BOUNCE_DEPTH>(
        const Ray & ray,
        Sphere * spheres,
        Light * lights,
        const Camera camera,
        const int sphereCount,
        const int lightCount ) {

    Vector3 color;
    return color;
}



__global__ void raytrace(
        Vector3 * image,
        Sphere * spheres,
        Light * lights,
        const Camera camera,
        const int sphereCount,
        const int lightCount ) {

    int width = camera.resolution.x;
    int height = camera.resolution.y;

    int idx = threadIdx.x
            + threadIdx.y * blockDim.y
            + blockIdx.x * blockDim.x * blockDim.y;

    if( idx >= width * height ) return;

    int i = idx % width;
    int j = height - idx / height;

    Ray ray = camera.castRayFromPixel( i, j );
    ray.direction.normalize();

    image[idx] = trace< 0 >( ray, spheres, lights, camera, sphereCount, lightCount );

}

__global__ void tonemap( Vector3 * image, float maxLuminance, Camera camera ) {

    int width = camera.resolution.x;
    int height = camera.resolution.y;

    int idx = threadIdx.x
            + threadIdx.y * blockDim.y
            + blockIdx.x * blockDim.x * blockDim.y;

    if( idx >= width * height ) return;

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

            sphere.material.albedo      = Vector3(1, 0.6, 0.8);
            sphere.material.roughness   = (float)i/(float)steps;
            sphere.material.metalness   = (float)j/(float)steps;
            sphere.material.opacity     = 1;

            scene.addSphere( sphere );
        }
    }

    {
        Sphere sphere;
        sphere.center   = Vector3( 150, 150, 500);
        sphere.size     = 130;

        sphere.material.albedo      = Vector3(1.0);
        sphere.material.roughness   = 0.5;
        sphere.material.metalness   = 0;

        scene.addSphere( sphere );
    }

    {
        Sphere sphere;
        sphere.center   = Vector3( 300, 300, 150);
        sphere.size     = 60;

        sphere.material.albedo      = Vector3(1.0);
        sphere.material.roughness   = 0.3;
        sphere.material.metalness   = 0;

        scene.addSphere( sphere );
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

class ArgParser {
    int argc = 0;
    char ** argv;

    int currentArg = 1;
public:
    ArgParser( int argc, char * argv[] )
        : argc(argc),argv(argv){}

    char * programCall() const {
        return argv[0];
    }

    void reset() {
        this->currentArg = 1;
    }

    int next() {
      return ++this->currentArg;
    }

    bool canGet() const {
        return this->currentArg < this->argc;
    }

    bool hasNext() const {
        return this->currentArg + 1 < this->argc;
    }

    char * getAndMove() {
        return this->argv[this->currentArg++];
    }

    char * current() const {
        return this->argv[this->currentArg];
    }

    int count() const {
        return argc-1;
    }

    int currentIdx() const {
        return currentArg;
    }

    int remaining() const {
        return count() - currentIdx();
    }
};

int main(int argc, char *argv[]) {

    auto argParser = ArgParser(argc,argv);

    Camera camera;
    camera.resolution = Vector3( 1024, 1024, 0) / 16  + Vector3(0,0,1);
    camera.size = Vector3( 300, 300, 1 );
    camera.position = Vector3( 150, 150, -100 );

    for( argParser.reset(); argParser.canGet(); argParser.next() ) {
        auto current = argParser.current();
        std::cout << "Parsing: " << current << " | Remaining: " << argParser.remaining() << std::endl;
        if( std::string(current) == "-res" && argParser.remaining() >= 2) {
            {
                std::stringstream ss;
                argParser.next();
                ss << argParser.current();
                ss >> camera.resolution.x;
            }
            {
                std::stringstream ss;
                argParser.next();
                ss << argParser.current();
                ss >> camera.resolution.y;
            }
            std::cout << "Resolution is set to " << camera.resolution << std::endl;
        }
    }

    cudaDeviceReset();

    Vector3 * imgData;

    Vector3 * d_imgData;
    Sphere * d_spheres;
    Light * d_lights;


    Scene scene;
    populateScene( scene );


    int width = camera.resolution.x;
    int height = camera.resolution.y;
    auto pixelCount = width * height;

    imgData = new Vector3[ pixelCount ];

    cudaMalloc( (void **) &d_imgData, pixelCount * sizeof( Vector3 ) );
    cudaMalloc( (void **) &d_spheres, scene.sphereCount * sizeof( Sphere ) );
    cudaMalloc( (void **) &d_lights, scene.lightCount * sizeof( Light ) );

    cudaMemcpy( d_spheres, scene.spheres, scene.sphereCount * sizeof( Sphere ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_lights, scene.lights, scene.lightCount * sizeof( Light ), cudaMemcpyHostToDevice );

    dim3 threaddim = dim3( 16, 16, 1 );
    auto totalBlockCount = ceil( pixelCount / (float)(threaddim.x * threaddim.y) );
    dim3 blockdim = dim3( totalBlockCount, 1, 1 );

    raytrace<<< blockdim, threaddim >>>( d_imgData, d_spheres, d_lights, camera, scene.sphereCount, scene.lightCount );

    CHECK_ERROR
    cudaMemcpy( imgData, d_imgData, pixelCount * sizeof( Vector3 ), cudaMemcpyDeviceToHost );
    writeP3( "output.ppm", imgData, width, height );


    // Apply tonemapping
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
        dim3 threaddim = dim3( 32, 1, 1 );
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
//            std::cout << "> " << i << std::endl;
//            std::cout << imgData[i] << " " << imgData[i+1] << std::endl;
//            std::cout << reduction[i] << " " << reduction[i+1] << std::endl;
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

    writeP3( "output_t.ppm", imgData, width, height );

    delete [] imgData;
}
