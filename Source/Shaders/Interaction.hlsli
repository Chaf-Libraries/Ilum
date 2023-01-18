#ifndef INTERATION_HLSLI
#define INTERATION_HLSLI

#include "Math.hlsli"

float3 OffsetRayOrigin(float3 p, float3 n)
{
    const float intScale = 256.0f;
    const float floatScale = 1.0f / 65536.0f;
    const float origin = 1.0f / 32.0f;

    int3 of_i = int3(intScale * n.x, intScale * n.y, intScale * n.z);

    float3 p_i = float3(asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                  asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                  asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return float3(abs(p.x) < origin ? p.x + floatScale * n.x : p_i.x, //
              abs(p.y) < origin ? p.y + floatScale * n.y : p_i.y, //
              abs(p.z) < origin ? p.z + floatScale * n.z : p_i.z);
}

struct Frame
{
    float3 x, y, z;
    
    void FromXZ(float3 x_, float3 z_)
    {
        x = x_;
        y = cross(z_, x_);
        z = z_;
    }
    
    void FromXY(float3 x_, float3 y_)
    {
        x = x_;
        y = y_;
        z = cross(x, y);
    }

    void FromZ(float3 z_)
    {
        z = z_;
        CoordinateSystem(z, x, y);
    }

    void FromX(float3 x_)
    {
        x = x_;
        CoordinateSystem(x, y, z);
    }

    void FromY(float3 y_)
    {
        y = y_;
        CoordinateSystem(y, z, x);
    }

    float3 ToLocal(float3 v)
    {
        return float3(dot(v, x), dot(v, y), dot(v, z));
    }

    float3 ToWorld(float3 v)
    {
        return v.x * x + v.y * y + v.z * z;
    }
};

struct Interaction
{
    float3 p;
    float3 n;
    float2 uv;
    float3 wo;
    float t;
    
    void Init(float3 p_, float3 n_, float2 uv_, float3 wo_, float t_)
    {
        p = p_;
        n = n_;
        uv = uv_;
        wo = wo_;
        t = t_;
    }
        
    RayDesc SpawnRay(float3 dir)
    {
        RayDesc ray;
        ray.Direction = dir;
        ray.Origin = OffsetRayOrigin(p, dot(dir, n) > 0.0 ? n : -n);
        return ray;
    }
    
    RayDesc SpawnRayTo(float3 position)
    {
        return SpawnRay(normalize(position - p));
    }
};

struct MediumInteraction
{
    Interaction isect;
};

struct SurfaceInteraction
{
    Interaction isect;
    
    /*float3 dpdu, dpdv;
    float3 dndu, dndv;
    
    struct
    {
        float3 n;
        float3 dpdu, dpdv;
        float3 dndu, dndv;
    } shading;*/
  
    float3 shading_n;
    uint material;
    float3 dpdx, dpdy;
    float3 dndx, dndy;
    float2 duvdx, duvdy;
};

struct VisibilityTester
{
    SurfaceInteraction from;
    float3 dir;
    float dist;
};

#endif 