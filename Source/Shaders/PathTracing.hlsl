#include "Interaction.hlsli"
#include "Math.hlsli"
#include "Common.hlsli"
#include "Material/BSDF/BSDF.hlsli"

RaytracingAccelerationStructure TopLevelAS;
ConstantBuffer<View> ViewBuffer;
RWTexture2D<float4> Output;

#ifndef RAYTRACING_PIPELINE
#define RAYGEN_SHADER
#define CLOSESTHIT_SHADER
#define MISS_SHADER
#include "Material/Material.hlsli"
#endif

struct PayLoad
{
    SurfaceInteraction interaction;

    float pdf;
    float3 wi;
    float4 color;
};

bool SceneIntersection(RayDesc ray, inout PayLoad pay_load)
{
    TraceRay(
        TopLevelAS, // RaytracingAccelerationStructure
        RAY_FLAG_FORCE_OPAQUE, // RayFlags
        0xFF, // InstanceInclusionMask
        0, // RayContributionToHitGroupIndex
        0, // MultiplierForGeometryContributionToHitGroupIndex
        0, // MissShaderIndex
        ray, // Ray
        pay_load // Payload
    );
    
    return pay_load.interaction.isect.t != Infinity;
}

#ifdef RAYGEN_SHADER
[shader("raygeneration")]
void RayGenMain()
{
    uint2 launch_index = DispatchRaysIndex().xy;
    uint2 launch_dims = DispatchRaysDimensions().xy;
    
    const float2 pixel_pos = float2(launch_index.xy) + 0.5;
    const float2 scene_uv = pixel_pos / float2(launch_dims.xy) * 2.0 - 1.0;
    
    PayLoad pay_load;
    
    RayDesc ray = ViewBuffer.CastRay(scene_uv);
    pay_load.color = 0.f;
    float4 color = 0.f;
    
    if (SceneIntersection(ray, pay_load))
    {
        color = pay_load.color;
    }

    Output[int2(launch_index.x, launch_dims.y - launch_index.y)] = float4(color);
}
#endif

#ifdef CLOSESTHIT_SHADER
[shader("closesthit")]
void ClosesthitMain(inout PayLoad pay_load : SV_RayPayload, BuiltInTriangleIntersectionAttributes attributes)
{
    Material material;
    material.Init();
    
    RayDesc ray;
    ray.Direction = WorldRayDirection();
    ray.Origin = WorldRayOrigin();
    // pay_load.interaction.Init(ray, attributes);
        
    //Material material;
    //material.Init();
    
    //uint light_count = 0;
    //uint stride = 0;
    //LightBuffer.GetDimensions(light_count, stride);
    
    pay_load.interaction.isect.t = RayTCurrent();
    pay_load.color = float4(material.bsdf.Eval(0, 0, TransportMode_Radiance), 1.f);
    
    //float3 wo = pay_load.interaction.p - View.position;
    
    //for (uint i = 0; i < 1; i++)
    //{
    //    float3 wi = 0.f;
    //   //pay_load.color += LightBuffer[i].Li(pay_load.interaction.p, wi) * material.SurfaceBSDFEval(wi, wo);
    //    //pay_load.color += LightBuffer[i].color;
    //
    //}
}
#endif

#ifdef MISS_SHADER
[shader("miss")]
void MissMain(inout PayLoad pay_load : SV_RayPayload)
{
    pay_load.interaction.isect.t = Infinity;
}
#endif
