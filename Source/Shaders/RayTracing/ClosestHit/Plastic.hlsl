#define USE_Plastic

#include "../../RayTracing.hlsli"

[shader("closesthit")]
void main(inout RayPayload rayPayload : SV_RayPayload, BuiltInTriangleIntersectionAttributes Attributes)
{
    rayPayload.hitT = RayTCurrent();
    rayPayload.baryCoord = Attributes.barycentrics;
        
    RayDesc ray;
    ray.Direction = WorldRayDirection();
    ray.Origin = WorldRayOrigin();
    
    GetInteraction(rayPayload, ray);

    Material material;
    GetMaterial(rayPayload.isect, material, ray, InstanceIndex());

    BSDFs bsdf = CreatePlasticMaterial(material);
    bsdf.isect = rayPayload.isect;
    
    rayPayload.f = bsdf.Samplef(rayPayload.isect.wo, rayPayload.rnd, rayPayload.wi, rayPayload.pdf, BSDF_ALL, rayPayload.sampled_type);
    rayPayload.emission = material.emissive;
}
