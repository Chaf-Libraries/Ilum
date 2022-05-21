#include "ShaderInterop.hpp"
#include "Common.hlsli"
#include "Math.hlsli"
#include "ShadingState.hlsli"
#include "Lights.hlsli"
#include "BxDF.hlsli"

Texture2D<uint> vbuffer : register(t0, space0);
RWTexture2D<float4> shading : register(u1, space0);
RWTexture2D<float2> normal : register(u2, space0);
cbuffer CameraBuffer : register(b3, space0)
{
    Camera camera;
}

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

[[vk::push_constant]]
struct
{
    uint directional_light_count;
    uint spot_light_count;
    uint point_light_count;
    uint area_light_count;
} push_constants;

struct Radiance
{
    float3 f_specular;
    float3 f_diffuse;
    float3 f_emissive;
    float3 f_clearcoat;
    float3 f_sheen;
    float3 f_transmission;
};

void GetPunctualRadiance(float3 iridescence_fresnel, float3 intensity, float3 L, float3 V, ShadingState sstate, inout Radiance radiance)
{
    float alpha_roughness = sstate.mat_info.roughness * sstate.mat_info.roughness;
    
    
    float3 H = normalize(L + V);
    
    float NoL = clamp(dot(sstate.normal, L), 0.0, 1.0);
    float NoV = clamp(dot(sstate.normal, V), 0.0, 1.0);
    float NoH = clamp(dot(sstate.normal, H), 0.0, 1.0);
    float LoH = clamp(dot(L, H), 0.0, 1.0);
    float VoH = clamp(dot(V, H), 0.0, 1.0);
        
    if (NoL > 0.0 || NoV > 0.0)
    {
        if (all(iridescence_fresnel != 0))
        {
            // Use Iridescence
            radiance.f_diffuse += intensity * NoL * Eval_BRDF_LambertianIridescence(sstate.mat_info.F0, sstate.mat_info.F90, iridescence_fresnel, sstate.mat_info.iridescence, sstate.mat_info.c_diff, sstate.mat_info.specular_weight, VoH);
            radiance.f_specular += intensity * NoL * Eval_BRDF_SpecularGGXIridescence(sstate.mat_info.F0, sstate.mat_info.F90, iridescence_fresnel, alpha_roughness, sstate.mat_info.iridescence, sstate.mat_info.specular_weight, VoH, NoL, NoV, NoH);
        }
        else
        {
            radiance.f_diffuse += intensity * NoL * Eval_BRDF_Lambertian(sstate.mat_info.F0, sstate.mat_info.F90, sstate.mat_info.c_diff, sstate.mat_info.specular_weight, VoH);
            radiance.f_specular += intensity * NoL * Eval_BRDF_SpecularGGX(sstate.mat_info.F0, sstate.mat_info.F90, alpha_roughness, sstate.mat_info.specular_weight, VoH, NoL, NoV, NoH);
        }
        
        // Sheen Term
        radiance.f_sheen += intensity * GetPunctualRadianceSheen(sstate.mat_info.sheen_color, sstate.mat_info.sheen_roughness, NoL, NoV, NoH);
            // TODO: Albedo Sheen Scaling
            //albedo_sheen_scaling = min(1.0 - max(sstate.mat_info.sheen_color.r, max(sstate.mat_info.sheen_color.g, sstate.mat_info.sheen_color.b)) * )

        // Clear Coat Term
        radiance.f_clearcoat += intensity * GetPunctualRadianceClearCoat(sstate.mat_info.clearcoat_normal, V, L, H, VoH, sstate.mat_info.clearcoatF0, sstate.mat_info.clearcoatF90, sstate.mat_info.clearcoat_roughness);
        
        // TODO: Transmission
        // TODO: Volume
    }
}

[numthreads(32, 32, 1)]
void main(CSParam param)
{
    uint2 extent;
    shading.GetDimensions(extent.x, extent.y);

    if (param.DispatchThreadID.x > extent.x || param.DispatchThreadID.y > extent.y)
    {
        return;
    }
    
    float2 screen_texcoord = (float2(param.DispatchThreadID.xy) + float2(0.5, 0.5)) / float2(extent);
    uint vdata = vbuffer[param.DispatchThreadID.xy];
    
    if (vdata == 0xffffffff)
    {
        shading[param.DispatchThreadID.xy] = float4(0.0, 0.0, 0.0, 1.0);
        normal[param.DispatchThreadID.xy] = PackNormal(float3(0.0, 0.0, 0.0));
        return;
    }
    
    ShadingState sstate;
    sstate.LoadVisibilityBuffer(vbuffer, param.DispatchThreadID.xy, camera);
    
    float3 V = normalize(camera.position - sstate.position);
    float NoV = clamp(dot(sstate.normal, V), 0.0, 1.0);
    
    float reflectance = max(max(sstate.mat_info.F0.r, sstate.mat_info.F0.g), sstate.mat_info.F0.b);
    
    Radiance radiance = (Radiance) 0;
    
    float albedo_sheen_scaling = 1.0;
    
    // Iridescence term
    float3 iridescence_fresnel = sstate.mat_info.F0;
    float3 iridescence_F0 = sstate.mat_info.F0;
    if (sstate.mat_info.iridescence_thickness == 0.0)
    {
        sstate.mat_info.iridescence = 0.0;
    }
    if (sstate.mat_info.iridescence > 0.0)
    {
        iridescence_fresnel = Iridescent_Fresnel(1.0, sstate.mat_info.iridescence_ior, NoV, sstate.mat_info.iridescence_thickness, sstate.mat_info.F0);
        iridescence_F0 = Schlick_to_F0(iridescence_fresnel, NoV);
    }
    
    // TODO: IBL
    // Iridescence
    // Clearcoat
    // Sheen
    // Volume Refraction
    
    // TODO: AO
    // AO Texture
    // SSAO
    // GTAO
    // RTAO
    float3 result;
    for (uint i = 0; i < push_constants.directional_light_count; i++)
    {
        float3 L;
        DirectionalLight light = directional_light[i];
        float3 intensity = Eval_Light(light, sstate.position, L);
        GetPunctualRadiance(iridescence_fresnel, intensity, L, V, sstate, radiance);
    }
    for (i = 0; i < push_constants.spot_light_count; i++)
    {
        float3 L;
        SpotLight light = spot_light[i];
        float3 intensity = Eval_Light(light, sstate.position, L);
        result = intensity;
        GetPunctualRadiance(iridescence_fresnel, intensity, L, V, sstate, radiance);
    }
    for (i = 0; i < push_constants.point_light_count; i++)
    {
        float3 L;
        PointLight light = point_light[i];
        float3 intensity = Eval_Light(light, sstate.position, L);
        result = intensity;
        GetPunctualRadiance(iridescence_fresnel, intensity, L, V, sstate, radiance);
    }
    
    radiance.f_emissive = sstate.mat_info.emissive;
    
    // Layer blending
    float clearcoat = sstate.mat_info.clearcoat;
    float3 clearcoat_fresnel = F_Schlick(sstate.mat_info.clearcoatF0, sstate.mat_info.clearcoatF90, clamp(dot(sstate.mat_info.clearcoat_normal, V), 0.0, 1.0));
    radiance.f_clearcoat *= clearcoat;
    
    float3 color = radiance.f_emissive + radiance.f_diffuse + radiance.f_specular + radiance.f_sheen;
    color = color * (1.0 - clearcoat * clearcoat_fresnel) + radiance.f_clearcoat;
    
    shading[param.DispatchThreadID.xy] = float4(color * 0.0000001 + sstate.mat_info.albedo.rgb, 1.0);
    normal[param.DispatchThreadID.xy] = PackNormal(sstate.normal.rgb);
}