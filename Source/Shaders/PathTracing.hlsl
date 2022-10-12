#include "Common.hlsli"

RaytracingAccelerationStructure TopLevelAS;
RWTexture2D<float4> OutputImage;

[shader("raygeneration")]
void RayGenMain()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDimensions = DispatchRaysDimensions().xy;
    
    const float2 pixelPos = float2(launchIndex.xy) + 0.5;
    const float2 sceneUV = pixelPos / float2(launchDimensions.xy) * 2.0 - 1.0;
    
    SurfaceInteraction interaction;
    
    RayDesc ray = View.CastRay(sceneUV);
    
    float3 color = 0.f;
    
    if (SceneIntersection(ray, TopLevelAS, interaction))
    {
        color = interaction.p;
    }

    OutputImage[int2(launchIndex.x, launchDimensions.y - launchIndex.y)] = float4(color, 1.0f);
}

[shader("closesthit")]
void ClosesthitMain(inout SurfaceInteraction interaction : SV_RayPayload, BuiltInTriangleIntersectionAttributes attributes)
{
    RayDesc ray;
    ray.Direction = WorldRayDirection();
    ray.Origin = WorldRayOrigin();
    interaction.Init(ray, attributes);
}

[shader("miss")]
void MissMain(inout SurfaceInteraction interaction : SV_RayPayload)
{
    interaction.t = Infinity;
}