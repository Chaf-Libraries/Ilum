#include "ShaderInterop.hpp"
#include "Common.hlsli"

float3 OffsetRay(float3 p, float3 n)
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

RayDesc SpawnRay(float3 position, float3 normal, float3 dir)
{
    RayDesc ray;
    ray.Direction = dir;
    ray.Origin = OffsetRay(position, dot(dir, normal) > 0.0 ? normal : -normal);
    return ray;
}

bool Intersection(AABB aabb, RayDesc ray, out float t)
{
    float3 inv_dir = rcp(ray.Direction);
    float t1 = (aabb.min_val.x - ray.Origin.x) * inv_dir.x;
    float t2 = (aabb.max_val.x - ray.Origin.x) * inv_dir.x;
    float t3 = (aabb.min_val.y - ray.Origin.y) * inv_dir.y;
    float t4 = (aabb.max_val.y - ray.Origin.y) * inv_dir.y;
    float t5 = (aabb.min_val.z - ray.Origin.z) * inv_dir.z;
    float t6 = (aabb.max_val.z - ray.Origin.z) * inv_dir.z;
    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));
    if (tmax < 0 || tmin > tmax)
    {
        return false;
    }
    
    t = tmin > 0 ? tmin : tmax;
    return true;
}

// Moller¨CTrumbore intersection
bool Intersection(float3 v0, float3 v1, float3 v2, RayDesc ray, out float t, out float3 bary)
{
    float eps = 1e-7;
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 h = cross(ray.Direction, edge2);
    float a = dot(edge1, h);
    // The ray is parallel to this triangle
    if (abs(a) < eps)
    {
        return false;
    }
    
    float f = 1.0 / a;
    float3 s = ray.Origin - v0;
    float u = f * dot(s, h);
    
    if (u < 0.0 || u > 1.0)
    {
        return false;
    }
    
    float3 q = cross(s, edge1);
    float v = f * dot(ray.Direction, q);
    
    if (v < 0.0 || u + v > 1.0)
    {
        return false;
    }
    
    t = f * dot(edge2, q);
    bary = float3(1.0 - u - v, u, v);
    
    return t > eps;
}