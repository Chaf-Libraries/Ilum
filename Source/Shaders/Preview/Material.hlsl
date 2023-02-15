#ifndef USE_MATERIAL
#include "../Material/Material.hlsli"
#endif

#ifndef RUNTIME
#define MATERIAL_ID 0
#endif

struct VSInput
{
    float3 Position : POSITION0;
    float3 Normal : NORMAL0;
    float3 Tangent : TANGENT0;
    float2 Texcoord0 : TEXCOORD0;
    float2 Texcoord1 : TEXCOORD1;
    uint InstanceID : SV_InstanceID;
};

struct VSOutput
{
    float4 Position_ : SV_Position;
    float3 Position : POSITION0;
    float3 Tangent : TANGENT0;
    float3 Normal : NORMAL0;
    float2 Texcoord : TEXCOORD0;
};

struct PSInput
{
    float3 Position : POSITION0;
    float3 Tangent : TANGENT0;
    float3 Normal : NORMAL0;
    float2 Texcoord : TEXCOORD0;
};

struct UniformBlock
{
    float4x4 transform;
    float4x4 model;
};

ConstantBuffer<UniformBlock> UniformBuffer;

VSOutput VSmain(VSInput input)
{
    VSOutput output = (VSOutput) 0;
    output.Position_ = mul(UniformBuffer.transform, mul(UniformBuffer.model, float4(input.Position, 1.f)));
    output.Position = output.Position_.xyz / output.Position_.w;
    output.Normal = normalize(mul((float3x3) UniformBuffer.model, input.Normal));
    output.Tangent = normalize(mul((float3x3) UniformBuffer.model, input.Tangent));
    output.Texcoord = input.Texcoord0;
    return output;
}

float4 PSmain(PSInput input) : SV_TARGET
{
    SurfaceInteraction surface_interaction;
    surface_interaction.isect.p = input.Position;
    surface_interaction.isect.n = input.Normal;
    surface_interaction.isect.n = input.Normal;
    surface_interaction.isect.nt = input.Tangent;
    surface_interaction.isect.uv = input.Texcoord;
    surface_interaction.duvdx = ddx(input.Texcoord);
    surface_interaction.duvdy = ddy(input.Texcoord);
    surface_interaction.material = MATERIAL_ID;
    
    Material material;
    material.Init(surface_interaction);
    
    float3 wo = normalize(float3(0.f, 2.f, 3.46f) - surface_interaction.isect.p);
    float3 wi = normalize(float3(0.f, 1.f, 3.f) - surface_interaction.isect.p);
    float intensity = 1.5f;
    
    return float4((material.bsdf.Eval(wo, wi, TransportMode_Radiance) * abs(dot(wo, input.Normal)) + material.bsdf.GetEmissive()) * intensity, 1.f);
}