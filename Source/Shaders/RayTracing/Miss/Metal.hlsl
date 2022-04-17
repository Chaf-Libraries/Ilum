#define USE_Metal

#include "../../RayTracing.hlsli"

[shader("miss")]
void main(inout RayPayload ray_payload : SV_RayPayload)
{
    ray_payload.visibility = true;
    
    RayDesc ray;
    ray.Direction = WorldRayDirection();
    ray.Origin = WorldRayOrigin();
    
    Material material;
    GetMaterial(ray_payload.isect, material, ray, ray_payload.material_idx);
    
    BSDFs bsdf = CreateMetalMaterial(material);
    bsdf.isect = ray_payload.isect;
    ray_payload.f = bsdf.f(ray_payload.isect.wo, ray_payload.wi, BSDF_ALL);
    ray_payload.pdf = bsdf.Pdf(ray_payload.isect.wo, ray_payload.wi, BSDF_ALL);
    ray_payload.bxdf_flags = bsdf.GetComponents();
    ray_payload.eta = 1.0;
}