#ifndef __RANDOM_HLSL__
#define __RANDOM_HLSL__

#include "Constants.hlsli"

struct PCGSampler
{
    uint seed;
    
    void Init(uint2 resolution, uint2 screen_coord, uint frame)
    {
        uint v0 = screen_coord.y * resolution.x + screen_coord.x;
        uint v1 = frame;
        uint s0 = 0;

        for (uint n = 0; n < 16; n++)
        {
            s0 += 0x9e3779b9;
            v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
            v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
        }
        
        seed = v0;
    }
    
    float Get1D()
    {
        // https://www.pcg-random.org/
        uint prev = seed * 747796405u + 2891336453u;
        uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
        seed = prev;
        uint r = (word >> 22u) ^ word;
        return asfloat(0x3f800000 | (r >> 9)) - 1.0f;
    }
    
    float2 Get2D()
    {
        float x = Get1D();
        float y = Get1D();
        return float2(x, y);
    }
    
    float3 Get3D()
    {
        float x = Get1D();
        float y = Get1D();
        float z = Get1D();
        return float3(x, y, z);
    }
};

float Rand1To1(float x)
{
    return frac(sin(x) * 43758.5453123);
}

float Rand2To1(float2 uv)
{
    return frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453);
}

float Rand3To1(float3 uvw)
{
    return frac(sin(dot(uvw, float3(12.9898, 78.233, 144.7272))) * 43758.5453);
}

float2 PoissonDiskSamples2D(float2 seed, int samples_num, int rings_num, int step)
{
    float angle = Rand2To1(seed) * PI * 2.0 + step * PI * 2.0 * float(rings_num) / float(samples_num);
    float radius = (float(step) + 1.0) / float(samples_num);
    return float2(cos(angle) * sin(angle), cos(angle)) * pow(radius, 0.75);
}

float3 PoissonDiskSamples3D(float3 seed, int samples_num, int rings_num, float2 step)
{
    float2 angle = Rand3To1(seed) * PI * 2.0 + step * PI * 2.0 * float(rings_num) / float(samples_num);
    float radius = (length(step) + 1.0) / float(samples_num);
    return float3(sin(angle.x) * cos(angle.y), sin(angle.x) * sin(angle.y), cos(angle.x)) * pow(radius, 0.75);
}

float2 UniformDiskSamples2D(float2 seed)
{
    float rand_num = Rand2To1(seed);
    float sample_x = Rand1To1(rand_num);
    float sample_y = Rand1To1(sample_x);

    float radius = sqrt(sample_x);
    float angle = sample_y * PI * 2;

    return float2(radius * cos(angle), radius * sin(angle));
}

float3 UniformDiskSamples3D(float3 seed)
{
    float sample_x = Rand3To1(seed);
    float sample_y = Rand1To1(sample_x);
    float sample_z = Rand1To1(sample_y);

    float radius = sqrt(sample_x);
    float2 angle = float2(sample_y * PI * 2, sample_z * PI * 2);

    return float3(sin(angle.y) * cos(angle.x), sin(angle.y) * sin(angle.x), cos(angle.y)) * radius;
}

float RadicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

float2 Hammersley(uint i, uint N)
{
    return float2(float(i) / float(N), RadicalInverse_VdC(i));
}

#endif