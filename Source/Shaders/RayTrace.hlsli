#include "ShaderInterop.hpp"
#include "Common.hlsli"

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
    
    t = tmin;
    return true;
}