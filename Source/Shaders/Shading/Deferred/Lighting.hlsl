#include "../../Common.hlsli"
#include "../../Material.hlsli"
#include "../../Light.hlsli"

#define LOCAL_SIZE 32

SamplerState texSampler : register(s0);
SamplerState shadowSampler : register(s1);
Texture2D GBuffer0 : register(t2);
Texture2D GBuffer1 : register(t3);
Texture2D GBuffer2 : register(t4);
Texture2D GBuffer3 : register(t5);
Texture2D GBuffer4 : register(t6);
Texture2D GBuffer5 : register(t7);
Texture2D DepthBuffer : register(t8);
Texture2D EmuLut : register(t9);
Texture2D EavgLut : register(t10);
Texture2DArray ShadowMaps : register(t11);
Texture2DArray CascadeShadowMaps : register(t12);
TextureCubeArray OmniShadowMaps : register(t13);
StructuredBuffer<DirectionalLight> directional_lights : register(t14);
StructuredBuffer<PointLight> point_lights : register(t15);
StructuredBuffer<SpotLight> spot_lights : register(t16);
ConstantBuffer<Camera> camera : register(b17);
Texture2D IrradianceSH : register(t18);
TextureCube PrefilterMap : register(t19);
Texture2D BRDFPreIntegrate : register(t20);
RWTexture2D<float4> Lighting : register(u21);

[[vk::push_constant]]
struct
{
    uint directional_light_count;
    uint spot_light_count;
    uint point_light_count;
    uint enable_multi_bounce;
    uint2 extent;
} push_constants;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

struct SH9
{
    float weights[9];
};

struct SH9Color
{
    float3 weights[9];
};

float LinearizeDepth(float depth, float znear, float zfar)
{
    float z = depth * 2.0 - 1.0;
    return znear * zfar / (zfar + depth * (znear - zfar));
}

float3 WorldPositionFromDepth(float2 uv, float depth, float4x4 view_projection_inverse)
{
    uv.y = 1.0 - uv.y;
    float2 screen_pos = uv * 2.0 - 1.0;
    float4 ndc_pos = float4(screen_pos, depth, 1.0);
    float4 world_pos = mul(view_projection_inverse, ndc_pos);
    world_pos = world_pos / world_pos.w;
    return world_pos.xyz;
}

[numthreads(LOCAL_SIZE, LOCAL_SIZE, 1)]
void main(CSParam param)
{
    if (param.DispatchThreadID.x > push_constants.extent.x || param.DispatchThreadID.y > push_constants.extent.y)
    {
        return;
    }
    
    float2 uv = (float2(param.DispatchThreadID.xy) + float2(0.5, 0.5)) / float2(push_constants.extent);
    float4 gbuffer0 = GBuffer0.SampleLevel(texSampler, uv, 0.0); // GBuffer 0: RGB - Albedo, A - Anisotropic
	float4 gbuffer1 = GBuffer1.SampleLevel(texSampler, uv, 0.0); // GBuffer 1: RGB - Normal, A - LinearDepth
	float4 gbuffer2 = GBuffer2.SampleLevel(texSampler, uv, 0.0); // GBuffer 2: R - Metallic, G - Roughness, B - Subsurface, A - EntityID
	float4 gbuffer3 = GBuffer3.SampleLevel(texSampler, uv, 0.0); // GBuffer 3: R - Sheen, G - Sheen Tint, B - Clearcoat, A - Clearcoat Gloss
	float4 gbuffer4 = GBuffer4.SampleLevel(texSampler, uv, 0.0); // GBuffer 4: RG - Velocity, B - Specular, A - Specular Tint
    float4 gbuffer5 = GBuffer5.SampleLevel(texSampler, uv, 0.0); // GBuffer 5: RGB - Emissive, A - Material Type
        
    float depth = DepthBuffer.SampleLevel(texSampler, uv, 0.0).r;

    Material material;
    material.base_color = float4(gbuffer0.rgb, 1.0);
    material.emissive = gbuffer5.rgb;
    material.subsurface = gbuffer2.b;
    material.metallic = gbuffer2.r;
    material.specular = gbuffer4.b;
    material.specular_tint = gbuffer4.a;
    material.roughness = clamp(gbuffer2.g, 0.01, 0.99);
    material.anisotropic = gbuffer0.a;
    material.sheen = gbuffer3.r;
    material.sheen_tint = gbuffer3.g;
    material.clearcoat = gbuffer3.b;
    material.clearcoat_gloss = gbuffer3.a;
    material.material_type = uint(gbuffer5.a);

    Lighting[int2(param.DispatchThreadID.xy)] = float4(material.base_color.rgb, 1.0);
}