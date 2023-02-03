#include "Interaction.hlsli"
#include "Math.hlsli"
#include "Common.hlsli"
#include "Light.hlsli"
#include "Random.hlsli"

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
StructuredBuffer<PointLight> PointLightBuffer;
StructuredBuffer<SpotLight> SpotLightBuffer;
StructuredBuffer<DirectionalLight> DirectionalLightBuffer;
StructuredBuffer<RectLight> RectLightBuffer;
ConstantBuffer<LightInfo> LightInfoBuffer;
RWTexture2D<float4> Output;

#ifndef RAYTRACING_PIPELINE
#define RAYGEN_SHADER
#define CLOSESTHIT_SHADER
#define MISS_SHADER
#include "Material/Material.hlsli"
#endif

#ifdef USE_SKYBOX
TextureCube<float4> Skybox;
SamplerState SkyboxSampler;
#endif

struct PayLoad
{
    SurfaceInteraction interaction;
    
    RayDiff ray_diff;

    PCGSampler rng;
    
    float3 wi;
    float pdf;
    float eta;
    float3 radiance;
    float3 throughout;
    bool terminate;
    bool visible;
    
    void Init()
    {
        radiance = 0.f;
        pdf = 1.f;
        eta = 1.f;
        throughout = 1.f;
        terminate = false;
    }
};

struct Light
{
    PointLight point_light;
    SpotLight spot_light;
    DirectionalLight directional_light;
    RectLight rect_light;
    uint light_type;
    
    void Init(uint light_id)
    {
        uint point_light_offset = 0;
        uint spot_light_offset = LightInfoBuffer.point_light_count;
        uint directional_light_offset = spot_light_offset + LightInfoBuffer.spot_light_count;
        uint rectangle_light_offset = directional_light_offset + LightInfoBuffer.directional_light_count;
        uint light_count = rectangle_light_offset + LightInfoBuffer.rect_light_count;
        
        if (light_id < spot_light_offset)
        {
            light_type = POINT_LIGHT;
            point_light = PointLightBuffer[light_id];
            return;
        }
        else if (light_id < directional_light_offset)
        {
            light_type = SPOT_LIGHT;
            spot_light = SpotLightBuffer[light_id - spot_light_offset];
            return;
        }
        else if (light_id < rectangle_light_offset)
        {
            light_type = DIRECTIONAL_LIGHT;
            directional_light = DirectionalLightBuffer[light_id - directional_light_offset];
            return;
        }
        else if (light_id < light_count)
        {
            light_type = RECT_LIGHT;
            point_light = PointLightBuffer[light_id - rectangle_light_offset];
            return;
        }
    }
    
    LightLiSample SampleLi(LightSampleContext ctx, float2 u)
    {
        switch (light_type)
        {
            case POINT_LIGHT:
                return point_light.SampleLi(ctx, u);
            case SPOT_LIGHT:
                return spot_light.SampleLi(ctx, u);
            case DIRECTIONAL_LIGHT:
                return directional_light.SampleLi(ctx, u);
        }
        
        LightLiSample light_sample;
        return light_sample;
    }

    float PDF_Li(LightSampleContext ctx, float3 wi)
    {
        switch (light_type)
        {
            case POINT_LIGHT:
                return point_light.PDF_Li(ctx, wi);
            case SPOT_LIGHT:
                return spot_light.PDF_Li(ctx, wi);
        }
        
        return 0;
    }
    
    bool IsDelta()
    {
        switch (light_type)
        {
            case POINT_LIGHT:
                return point_light.IsDelta();
            case SPOT_LIGHT:
                return spot_light.IsDelta();
            case DIRECTIONAL_LIGHT:
                return directional_light.IsDelta();
            case RECT_LIGHT:
                return rect_light.IsDelta();
        }
        return false;
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

bool Unoccluded(SurfaceInteraction surface_interaction, Interaction isect)
{
    float3 direction = normalize(isect.p - surface_interaction.isect.p);
    RayDesc shadow_ray = surface_interaction.isect.SpawnRay(direction);
    shadow_ray.TMin = 0.0;
    shadow_ray.TMax = distance(isect.p, surface_interaction.isect.p);
    
    PayLoad pay_load;
    pay_load.visible = true;
    
    TraceRay(
        TopLevelAS, // RaytracingAccelerationStructure
        RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, // RayFlags
        0xFF, // InstanceInclusionMask
        0, // RayContributionToHitGroupIndex
        1, // MultiplierForGeometryContributionToHitGroupIndex
        0, // MissShaderIndex
        shadow_ray, // Ray
        pay_load // Payload
    );
    
    return !pay_load.visible;
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
#ifdef USE_SKYBOX
            // Environment Sampling (HDR)
            // See:  https://arxiv.org/pdf/1901.05423.pdf
            float pdf = 1.0;
            float3 wi = normalize(ray.Direction);
            pay_load.radiance += float4(Skybox.SampleLevel(SkyboxSampler, wi, 0.0).rgb * pay_load.throughout, 1.0);
#else
            float pdf = 1.0;
            //pay_load.radiance += 0.2f * pay_load.throughout;
#endif
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
    float lum = Luminance(pay_load.radiance);
    if (lum > ConfigBuffer.clamp_threshold)
    {
        pay_load.radiance *= ConfigBuffer.clamp_threshold / lum;
    }
    
    int2 launch_id = int2(launch_index.x, launch_dims.y - launch_index.y);
    
    // Temporal Accumulation
    if (ConfigBuffer.frame_count == 0)
    {
        Output[launch_id] = float4(pay_load.radiance, 1.f);
    }
    else
    {
        float3 prev_color = Output.Load(int3(launch_id.x, launch_id.y, 0)).rgb;
        float4 accumulated_color = 0.f;
        if ((isnan(prev_color.x) || isnan(prev_color.y) || isnan(prev_color.z)))
        {
            accumulated_color = float4(pay_load.radiance, 1.f);
        }
        else
        {
            accumulated_color = float4(lerp(prev_color, pay_load.radiance, 1.0 / float(ConfigBuffer.frame_count + 1)), 1.f);
        }
        
        Output[launch_id] = accumulated_color;
    }
}
#endif

#ifdef CLOSESTHIT_SHADER

float3 SampleLd(PayLoad pay_load, Material material)
{
    uint flags = material.bsdf.Flags();
    //if(IsReflective(flags)&&!IsTransmissive(flags))
    //{
    //    OffsetRayOrigin(pay_load.interaction.isect.wo, pay_load.interaction.isect.n);
    //}
    
    uint light_count = LightInfoBuffer.point_light_count + LightInfoBuffer.spot_light_count + LightInfoBuffer.directional_light_count + LightInfoBuffer.rect_light_count;
    
    // Uniformly sample one light
    if (light_count > 0)
    {
        uint light_id = (uint) min(light_count - 1, pay_load.rng.Get1D() * (float) light_count);
        float light_select_pdf = 1.0 / (float) light_count;
        
        Light light;
        light.Init(light_id);
        
        LightSampleContext sample_context;
        sample_context.n = pay_load.interaction.isect.n;
        sample_context.ns = pay_load.interaction.shading_n;
        sample_context.p = pay_load.interaction.isect.p;
        
        LightLiSample light_sample = light.SampleLi(sample_context, pay_load.rng.Get2D());
        
        float3 wo = pay_load.interaction.isect.wo;
        float3 f = material.bsdf.Eval(wo, light_sample.wi, TransportMode_Radiance) * abs(dot(light_sample.wi, pay_load.interaction.shading_n));
        
        if (IsBlack(light_sample.L) || IsBlack(f) || !Unoccluded(pay_load.interaction, light_sample.isect))
        {
            return 0.f;
        }
        
        float p_l = light_select_pdf * light_sample.pdf;
        
        if (light.IsDelta())
        {
            //return f;
            return light_sample.L * f / p_l;
        }
        else
        {
            float p_b = material.bsdf.PDF(wo, light_sample.wi, TransportMode_Radiance, BSDF_All);
            float w_l = PowerHeuristic(1, p_l, 1, p_b);
            return w_l * light_sample.L * f / p_l;
        }
    }
    return 0.f;
}
#include "Material/Scattering.hlsli"
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
    surface_interaction.shading_n = dot(surface_interaction.isect.n, -WorldRayDirection()) <= 0 ? -surface_interaction.isect.n : surface_interaction.isect.n;
    surface_interaction.isect.wo = -normalize(WorldRayDirection());
    surface_interaction.isect.t = RayTCurrent();
    surface_interaction.material = instance.material_id;
    
    float3 edge01 = mul(v1.position - v0.position, (float3x3) ObjectToWorld4x3()).xyz;
    float3 edge02 = mul(v2.position - v0.position, (float3x3) ObjectToWorld4x3()).xyz;
    float3 wgeom_normal = cross(edge01, edge02);
    
    // Compute Differential
    pay_load.ray_diff.Propagate(normalize(WorldRayDirection()), RayTCurrent(), surface_interaction.shading_n);
    
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
    
    ///////////////////////////
   //Frame frame;
   //frame.FromZ(surface_interaction.isect.n);
   //pay_load.radiance = float4(frame.z, 1.f);
   //pay_load.terminate = true;
   //return;
    //Frame frame;
    //frame.FromZ(surface_interaction.isect.n);
    //TrowbridgeReitzDistribution distrib;
    //distrib.Init(distrib.RoughnessToAlpha(0.9), distrib.RoughnessToAlpha(0.9));
    //float3 wo_ = frame.ToLocal(surface_interaction.isect.wo);
    //pay_load.radiance = float4(distrib.Sample_wm(wo_, pay_load.rng.Get2D()), 1.f);
    //pay_load.terminate = true;
    //return;
    ////////////////////////
    
    Material material;
    material.Init(surface_interaction);
    
    pay_load.interaction = surface_interaction;
    float3 wo = surface_interaction.isect.wo;
    
    if (IsNonSpecular(material.bsdf.Flags()))
    {
        uint flags = material.bsdf.Flags();
        float3 Ld = SampleLd(pay_load, material);
        pay_load.radiance = pay_load.radiance.rgb + pay_load.throughout * Ld;
    }
    
    // Sample BSDF to get new path direction
    uint sample_type;
    BSDFSample bs = material.bsdf.Samplef(wo, pay_load.rng.Get1D(), pay_load.rng.Get2D(), TransportMode_Radiance, BSDF_All);
   
    if (IsBlack(bs.f) || bs.pdf == 0.0)
    {
        pay_load.terminate = true;
        return;
    }
    
    pay_load.throughout *= bs.f * abs(dot(bs.wiW, pay_load.interaction.shading_n)) / bs.pdf;
    pay_load.pdf = bs.pdfIsProportional ? material.bsdf.PDF(wo, bs.wiW, TransportMode_Radiance, BSDF_All) : bs.pdf;
    bool specular_bounce = bs.IsSpecular();
    if (bs.IsTransmission())
    {
        pay_load.eta *= Sqr(bs.eta);
    }
    pay_load.wi = bs.wiW;
   //Frame frame;
 // frame.FromZ(surface_interaction.isect.n);
    //float3 a = dot(frame.y, frame.x);
    //pay_load.radiance = float4(bs.f, 1.f);
}
#endif

#ifdef MISS_SHADER
[shader("miss")]
void MissMain(inout PayLoad pay_load : SV_RayPayload)
{
    pay_load.visible = false;
}
#endif
