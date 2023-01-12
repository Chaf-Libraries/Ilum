#include "Interaction.hlsli"
#include "Math.hlsli"
#include "Common.hlsli"
#include "Light.hlsli"
#include "Random.hlsli"
#include "Material/BSDF/BSDF.hlsli"

struct Config
{
    uint max_bounce;
    uint max_spp;
    uint frame_count;
    float clamp_threshold;
};

RaytracingAccelerationStructure TopLevelAS;
StructuredBuffer<Instance> InstanceBuffer;
StructuredBuffer<Vertex> VertexBuffer[];
StructuredBuffer<uint> IndexBuffer[];
ConstantBuffer<View> ViewBuffer;
ConstantBuffer<Config> ConfigBuffer;
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
    
    RayDiff ray_diff;

    PCGSampler rng;
    
    float3 wi;
    float eta;
    float4 radiance;
    float3 throughout;
    bool terminate;
    bool visible;
    
    void Init()
    {
        radiance = 0.f;
        eta = 1.f;
        throughout = 1.f;
        terminate = false;
    }
};

bool SceneIntersection(RayDesc ray, out PayLoad pay_load)
{
    pay_load.visible = true;
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
    
    return pay_load.visible;
}

bool Unoccluded(inout PayLoad pay_load, VisibilityTester visibility)
{
    pay_load.interaction.isect.t = Infinity;
    RayDesc shadow_ray = pay_load.interaction.isect.SpawnRay(normalize(visibility.dir));
    shadow_ray.TMin = 0.0;
    shadow_ray.TMax = visibility.dist;
    
    PayLoad shadow_pay_load;
    shadow_pay_load.visible = true;
    
    TraceRay(
        TopLevelAS, // RaytracingAccelerationStructure
        RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, // RayFlags
        0xFF, // InstanceInclusionMask
        0, // RayContributionToHitGroupIndex
        1, // MultiplierForGeometryContributionToHitGroupIndex
        0, // MissShaderIndex
        shadow_ray, // Ray
        shadow_pay_load // Payload
    );
    
    return !shadow_pay_load.visible;
}

float Luminance(float3 color)
{
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f)); //color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;
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
    pay_load.rng.Init(launch_dims, launch_index, ConfigBuffer.frame_count);
    pay_load.Init();
    
    RayDesc ray;
    ViewBuffer.CastRay(scene_uv, (float2) launch_dims, ray, pay_load.ray_diff);
    
    for (uint bounce = 0; bounce < ConfigBuffer.max_bounce && !pay_load.terminate; bounce++)
    {
        if (!SceneIntersection(ray, pay_load))
        {
            // Sample environment light
            float pdf = 0;
            //ray_payload.radiance += ray_payload.throughout * EnvironmentSampling(ray_payload, normalize(ray.Direction), pdf);
            break;
        }
           
        ray = pay_load.interaction.isect.SpawnRay(pay_load.wi);
        
        // Russian roulette
        float3 rrBeta = pay_load.throughout * pay_load.eta;
        float max_cmpt = max(rrBeta.x, max(rrBeta.y, rrBeta.z));
        if (max_cmpt < 1.0 && bounce > 3)
        {
            float q = max(0.05, 1.0 - max_cmpt);
            if (pay_load.rng.Get1D() < q)
            {
                break;
            }
            pay_load.throughout /= 1.0 - q;
        }
    }
    
    // Clamp firefly
    float lum = Luminance(pay_load.radiance.rgb);
    if (lum > ConfigBuffer.clamp_threshold)
    {
        pay_load.radiance.rgb *= ConfigBuffer.clamp_threshold / lum;
    }
    
    int2 launch_id = int2(launch_index.x, launch_dims.y - launch_index.y);
    
    // Temporal Accumulation
    if (ConfigBuffer.frame_count == 0)
    {
        Output[launch_id] = pay_load.radiance;
    }
    else
    {
        float3 prev_color = Output[launch_id].rgb;
        float4 accumulated_color = 0.f;
        if ((isnan(prev_color.x) || isnan(prev_color.y) || isnan(prev_color.z)))
        {
            accumulated_color = pay_load.radiance;
        }
        else
        {
            accumulated_color = float4(lerp(prev_color, pay_load.radiance.rgb, 1.0 / float(ConfigBuffer.frame_count + 1)), pay_load.radiance.a);
        }

        Output[launch_id] = accumulated_color;
    }
    
    //float4 radiance = 0.f;
    //
    //if (SceneIntersection(ray, pay_load))
    //{
    //    radiance = float4(pay_load.radiance, 1.f);
    //}
    //
    //Output[int2(launch_index.x, launch_dims.y - launch_index.y)] = float4(radiance);
}
#endif

#ifdef CLOSESTHIT_SHADER
[shader("closesthit")]
void ClosesthitMain(inout PayLoad pay_load : SV_RayPayload, BuiltInTriangleIntersectionAttributes attributes)
{
    pay_load.visible = true;
    
    SurfaceInteraction surface_interaction;
    
    const uint instance_id = InstanceIndex();
    const uint primitve_id = PrimitiveIndex();
    const float3 bary = float3(1.0 - attributes.barycentrics.x - attributes.barycentrics.y, attributes.barycentrics.x, attributes.barycentrics.y);
    
    Instance instance = InstanceBuffer[instance_id];

    Vertex v0 = VertexBuffer[instance.mesh_id][IndexBuffer[instance.mesh_id][primitve_id * 3]];
    Vertex v1 = VertexBuffer[instance.mesh_id][IndexBuffer[instance.mesh_id][primitve_id * 3 + 1]];
    Vertex v2 = VertexBuffer[instance.mesh_id][IndexBuffer[instance.mesh_id][primitve_id * 3 + 2]];
    
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
    
    float3 edge01 = mul(v1.position - v0.position, (float3x3) ObjectToWorld4x3()).xyz;
    float3 edge02 = mul(v2.position - v0.position, (float3x3) ObjectToWorld4x3()).xyz;
    float3 wgeom_normal = cross(edge01, edge02);
    surface_interaction.geo_n = normalize(wgeom_normal);
    
    // Compute Differential
    pay_load.ray_diff.Propagate(normalize(WorldRayDirection()), RayTCurrent(), surface_interaction.isect.n);
    
    float3 nu = cross(edge02, wgeom_normal);
    float3 nv = cross(edge01, wgeom_normal);
    float3 lu = nu / (dot(nu, edge01));
    float3 lv = nv / (dot(nv, edge02));

    float2 dBarydx = float2(dot(lu, pay_load.ray_diff.dOdx), dot(lv, pay_load.ray_diff.dOdx));
    float2 dBarydy = float2(dot(lu, pay_load.ray_diff.dOdy), dot(lv, pay_load.ray_diff.dOdy));
    
    float2 deltaUV1 = v1.texcoord0 - v0.texcoord0;
    float2 deltaUV2 = v2.texcoord0 - v0.texcoord0;
    
    surface_interaction.duvdx = dBarydx.x * deltaUV1 + dBarydx.y * deltaUV2;
    surface_interaction.duvdy = dBarydy.x * deltaUV1 + dBarydy.y * deltaUV2;
    
    Material material;
    material.Init(surface_interaction);
    
    pay_load.interaction = surface_interaction;
   
    uint light_count, light_stride;
    LightInfo.GetDimensions(light_count, light_stride);
    light_count /= 2;
    
    Frame frame;
    frame.CreateCoordinateSystem(surface_interaction.isect.n);
        
    float3 wo = frame.ToLocal(surface_interaction.isect.wo);
    
    // Uniformly sample one light
    if (light_count > 0)
    {
        uint light_id = (uint) min(light_count - 1, pay_load.rng.Get1D() * (float) light_count);
        float light_select_pdf = 1.0 / (float) light_count;
        
        Light light;
        light.Init(light_id);
        
        VisibilityTester visibility;
        float light_sample_pdf, scattering_pdf;
        float3 Li = light.SampleLi(surface_interaction, pay_load.rng.Get2D(), pay_load.wi, light_sample_pdf, visibility);
        float3 Ld = float3(0.0, 0.0, 0.0);
        
        float3 wi = frame.ToLocal(pay_load.wi);
        
        if (!IsBlack(Li) && Unoccluded(pay_load, visibility) && light_sample_pdf != 0.0)
        {
            // Evaluate BSDF
            float3 f = material.bsdf.Eval(wo, wi, TransportMode_Radiance) * abs(dot(pay_load.wi, pay_load.interaction.isect.n));
            scattering_pdf = material.bsdf.PDF(wo, wi, TransportMode_Radiance, BSDF_All);
            // Add light's contribution to reflected radiance
            if (light.IsDelta())
            {
                Ld += f * Li / light_sample_pdf;
            }
            else
            {
                // Light MIS
                float weight = PowerHeuristic(1, light_sample_pdf, 1, scattering_pdf);
                Ld += f * Li * weight / light_sample_pdf;
            }
        }
        
        // Sample BSDF with MIS
        if (!light.IsDelta())
        {
            // TODO
        }
        
        pay_load.radiance += float4(Ld / light_select_pdf * pay_load.throughout, 1.0);
    }
    
    // Sample BSDF to get new path direction
    uint sample_type;
    BSDFSample bs = material.bsdf.Samplef(wo, pay_load.rng.Get1D(), pay_load.rng.Get2D(), TransportMode_Radiance, BSDF_All);
   
    if (IsBlack(bs.f) || bs.pdf == 0.0)
    {
        pay_load.terminate = true;
        return;
    }
    
    if (bs.IsTransmission())
    {
        pay_load.eta *= (dot(surface_interaction.isect.wo, surface_interaction.geo_n) > 0) ? (bs.eta * bs.eta) : 1 / (bs.eta * bs.eta);
    }
    
    float3 wiW = frame.ToWorld(bs.wi);
    
    if (bs.pdf > 0.0 && !IsBlack(bs.f) && abs(dot(wiW, pay_load.interaction.isect.n)) != 0.0)
    {
        pay_load.throughout *= bs.f * abs(dot(wiW, pay_load.interaction.isect.n)) / bs.pdf;
        pay_load.wi = wiW;
    }
    else
    {
        pay_load.terminate = true;
        return;
    }
}
#endif

#ifdef MISS_SHADER
[shader("miss")]
void MissMain(inout PayLoad pay_load : SV_RayPayload)
{
    pay_load.visible = false;
}
#endif
