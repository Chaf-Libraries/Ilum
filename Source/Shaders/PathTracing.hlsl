#include "Interaction.hlsli"
#include "Math.hlsli"
#include "Common.hlsli"
#include "Material/BSDF/BSDF.hlsli"

RaytracingAccelerationStructure TopLevelAS : register(t994);
StructuredBuffer<Instance> InstanceBuffer : register(t995);
StructuredBuffer<Vertex> VertexBuffer[] : register(t996);
StructuredBuffer<uint> IndexBuffer[] : register(t997);
ConstantBuffer<View> ViewBuffer : register(b998);
RWTexture2D<float4> Output : register(u999);

#ifndef RAYTRACING_PIPELINE
#define RAYGEN_SHADER
#define CLOSESTHIT_SHADER
#define MISS_SHADER
#include "Material/Material.hlsli"
#endif

struct PayLoad
{
    SurfaceInteraction interaction;
    
    RayDiff ray_diff;

    float pdf;
    float3 wi;
    float4 color;
};

bool SceneIntersection(RayDesc ray, out PayLoad pay_load)
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
    
    RayDesc ray;
    ViewBuffer.CastRay(scene_uv, ray, pay_load.ray_diff);
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
    SurfaceInteraction surface_interaction;
    
    const uint instance_id = InstanceIndex();
    const uint primitve_id = PrimitiveIndex();
    const float3 bary = float3(1.0 - attributes.barycentrics.x - attributes.barycentrics.y, attributes.barycentrics.x, attributes.barycentrics.y);
    
    Instance instance = InstanceBuffer[instance_id];

    Vertex v0 = VertexBuffer[instance_id][IndexBuffer[instance_id][primitve_id * 3]];
    Vertex v1 = VertexBuffer[instance_id][IndexBuffer[instance_id][primitve_id * 3 + 1]];
    Vertex v2 = VertexBuffer[instance_id][IndexBuffer[instance_id][primitve_id * 3 + 2]];
    
    surface_interaction.isect.p = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    surface_interaction.isect.uv = v0.texcoord0.xy * bary.x + v1.texcoord0.xy * bary.y + v2.texcoord0.xy * bary.z;
    surface_interaction.isect.n = v0.normal.xyz * bary.x + v1.normal.xyz * bary.y + v2.normal.xyz * bary.z;
    surface_interaction.isect.n = normalize(mul(WorldToObject4x3(), normalize(surface_interaction.isect.n)).xyz);
    if (dot(surface_interaction.isect.n, -WorldRayDirection()) <= 0)
    {
        surface_interaction.isect.n *= -1.0f;
    }
    surface_interaction.isect.wo = -normalize(WorldRayDirection());
    surface_interaction.isect.t = RayTCurrent();
    surface_interaction.material = instance.material_id;
    
    // Compute Differential
    pay_load.ray_diff.Propagate(normalize(WorldRayDirection()), RayTCurrent(), surface_interaction.isect.n);
    
    float3 geom_normal = normalize(cross(v2.position - v0.position, v1.position - v0.position));
    float3 wgeom_normal = normalize(mul(WorldToObject4x3(), geom_normal).xyz);
    
    float3 edge01 = mul((float3x3) WorldToObject4x3(), v1.position - v0.position);
    float3 edge02 = mul((float3x3) WorldToObject4x3(), v2.position - v0.position);
    
    float3 nu = cross(edge02, wgeom_normal);
    float3 nv = cross(edge01, wgeom_normal);
    float3 lu = nu / (dot(nu, edge01));
    float3 lv = nv / (dot(nv, edge02));

    float2 dBarydx = float2(dot(lu, pay_load.ray_diff.dOdx), dot(lu, pay_load.ray_diff.dOdy));
    float2 dBarydy = float2(dot(lv, pay_load.ray_diff.dOdx), dot(lv, pay_load.ray_diff.dOdy));
    
    float2 deltaUV1 = v1.texcoord0 - v0.texcoord0;
    float2 deltaUV2 = v2.texcoord0 - v0.texcoord0;
    
    surface_interaction.duvdx = dBarydx.x * deltaUV1 + dBarydx.y * deltaUV2;
    surface_interaction.duvdy = dBarydy.x * deltaUV1 + dBarydy.y * deltaUV2;
    
    Material material;
    material.Init(surface_interaction);
   
    pay_load.interaction = surface_interaction;
    pay_load.color = float4(material.bsdf.Eval(1, 1, TransportMode_Radiance), 1.f);

    
}
#endif

#ifdef MISS_SHADER
[shader("miss")]
void MissMain(inout PayLoad pay_load : SV_RayPayload)
{
    pay_load.interaction.isect.t = Infinity;
}
#endif
