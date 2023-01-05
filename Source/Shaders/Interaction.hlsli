#ifndef INTERATION_HLSLI
#define INTERATION_HLSLI

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
    float3 s, t;
    float3 n;
    
    void CreateCoordinateSystem(float3 normal)
    {
        n = normal;
        if (n.z < 0.f)
        {
            const float a = 1.0f / (1.0f - n.z);
            const float b = n.x * n.y * a;
            s = float3(1.0f - n.x * n.x * a, -b, n.x);
            t = float3(b, n.y * n.y * a - 1.0f, -n.y);
        }
        else
        {
            const float a = 1.0f / (1.0f + n.z);
            const float b = -n.x * n.y * a;
            s = float3(1.0f - n.x * n.x * a, b, -n.x);
            t = float3(b, 1.0f - n.y * n.y * a, -n.y);
        }
    }
    
    float3 ToLocal(float3 v)
    {
        return float3(dot(v, s), dot(v, t), dot(v, n));
    }
    
    float3 ToWorld(float3 v)
    {
        return n * v.z + (t * v.y + s * v.x);
    }
    
    float CosTheta2(float3 v)
    {
        return v.z * v.z;
    }
    
    float CosTheta(float3 v)
    {
        return v.z;
    }
    
    float SinTheta2(float3 v)
    {
        return v.x * v.x + v.y * v.z;
    }
    
    float SinTheta(float3 v)
    {
        return sqrt(max(SinTheta2(v), 0));
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
    
    float3 dpdu, dpdv;
    float3 dndu, dndv;
    
    struct
    {
        float3 n;
        float3 dpdu, dpdv;
        float3 dndu, dndv;
    } shading;
  
    uint material;
    float3 dpdx, dpdy;
    float dudx, dvdx, dudy, dvdy;
    
    void Init(float3 p_, float2 uv_, float3 wo_, float t_, float3 dpdu_, float3 dpdv_, float3 dndu_, float3 dndv_)
    {
        isect.Init(p_, normalize(cross(dpdu, dpdv)), uv_, wo_, t_);
        dpdu = dpdu_;
        dpdv = dpdv_;
        dndu = dndu_;
        dndv = dndv_;
        
        shading.dpdu = dpdu_;
        shading.dpdv = dpdv_;
        shading.dndu = dndu_;
        shading.dndv = dndv_;
        shading.n = isect.n;
    }
    
    void SetShadingGeometry(float3 ns, float3 dpdus, float3 dpdvs, float3 dndus, float3 dndvs, bool authoritative_orientation)
    {
        shading.n = ns;
        if (authoritative_orientation)
        {
            isect.n = (dot(isect.n, shading.n) < 0.f) ? -isect.n : isect.n;
        }
        else
        {
            shading.n = (dot(shading.n, isect.n) < 0.f) ? -shading.n : shading.n;
        }
        
        shading.dpdu = dpdus;
        shading.dpdv = dpdvs;
        shading.dndu = dndus;
        shading.dndv = dndvs;
        
        while (dot(shading.dpdu, shading.dpdu) > 1e16 ||
            dot(shading.dpdu, shading.dpdv) > 1e16)
        {
            shading.dpdu /= 1e8f;
            shading.dpdv /= 1e8f;
        }
    }
};

#endif 