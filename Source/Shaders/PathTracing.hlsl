#include "Common.hlsli"

#ifndef RUNTIME
#define RAYGEN_STAGE
#define CLOSESTHIT_STAGE
#define MISS_STAGE
#endif

RaytracingAccelerationStructure TopLevelAS;
RWTexture2D<float4> OutputImage;

struct PayLoad
{
    SurfaceInteraction interaction;
    float pdf;
    float3 wi;
    float3 color;
};

bool SceneIntersection(RayDesc ray, inout PayLoad pay_load)
{
    TraceRay(
        TopLevelAS, // RaytracingAccelerationStructure
        RAY_FLAG_NONE, // RayFlags
        0xFF, // InstanceInclusionMask
        0, // RayContributionToHitGroupIndex
        1, // MultiplierForGeometryContributionToHitGroupIndex
        0, // MissShaderIndex
        ray, // Ray
        pay_load // Payload
    );
    
    return pay_load.interaction.t != Infinity;
}

#ifdef RAYGEN_STAGE
[shader("raygeneration")]
void RayGenMain()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDimensions = DispatchRaysDimensions().xy;
    
    const float2 pixelPos = float2(launchIndex.xy) + 0.5;
    const float2 sceneUV = pixelPos / float2(launchDimensions.xy) * 2.0 - 1.0;
    
    PayLoad pay_load;
    
    RayDesc ray = View.CastRay(sceneUV);
    
    float3 color = 0.f;
    
    if (SceneIntersection(ray, pay_load))
    {
        color = pay_load.color;
    }

    OutputImage[int2(launchIndex.x, launchDimensions.y - launchIndex.y)] = float4(color, 1.0f);
}
#endif

#ifdef CLOSESTHIT_STAGE

#include "Material.hlsli"

[shader("closesthit")]
void ClosesthitMain(inout PayLoad pay_load : SV_RayPayload, BuiltInTriangleIntersectionAttributes attributes)
{
    RayDesc ray;
    ray.Direction = WorldRayDirection();
    ray.Origin = WorldRayOrigin();
    pay_load.interaction.Init(ray, attributes);
        
    Material material;
    material.Init();
    
    uint light_count = 0;
    uint stride = 0;
    LightBuffer.GetDimensions(light_count, stride);
    
    pay_load.color = 0.f;
    
    float3 wo = pay_load.interaction.p - View.position;
    
    for (uint i = 0; i < light_count; i++)
    {
        float3 wi = 0.f;
       pay_load.color += LightBuffer[i].Li(pay_load.interaction.p, wi) * material.SurfaceBSDFEval(wi, wo);
    }
}
#endif

#ifdef MISS_STAGE
[shader("miss")]
void MissMain(inout SurfaceInteraction interaction : SV_RayPayload)
{
    interaction.t = Infinity;
}
#endif