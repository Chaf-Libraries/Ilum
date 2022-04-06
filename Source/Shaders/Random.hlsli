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
        return asfloat(0x3f800000 | (((word >> 22u) ^ word) >> 9)) - 1.0f;
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

#endif