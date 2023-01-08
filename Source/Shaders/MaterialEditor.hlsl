#ifndef USE_MATERIAL
#include "Material/Material.hlsli"
#endif

struct VSInput
{
    float3 Position : POSITION0;
    float3 Normal : NORMAL0;
    float3 Tangent : TANGENT0;
    float2 Texcoord0 : TEXCOORD0;
    float2 Texcoord1 : TEXCOORD1;
};

struct UniformBlock
{
    float4x4 transform;
    float4x4 model;
};

ConstantBuffer<UniformBlock> UniformBuffer;

struct VSOutput
{
    float4 Position_ : SV_Position;
    float3 Position : POSITION0;
    float3 Normal : NORMAL0;
    float2 Texcoord : TEXCOORD0;
};

VSOutput VSmain(VSInput input)
{
    VSOutput output = (VSOutput) 0;
    output.Position_ = mul(UniformBuffer.transform, mul(UniformBuffer.model, float4(input.Position, 1.f)));
    output.Position = output.Position_.xyz/output.Position_.w;
    output.Normal = input.Normal;
    output.Texcoord = input.Texcoord0;
    return output;
}

struct PSInput
{
    float3 Position : POSITION0;
    float3 Normal : NORMAL0;
    float2 Texcoord : TEXCOORD0;
};

float4 PSmain(PSInput input) : SV_TARGET
{
    SurfaceInteraction surface_interaction;
    surface_interaction.Init(input.Position, input.Texcoord, 0, 0, 0, 0, 0, 0);
    surface_interaction.isect.n = input.Normal;
    surface_interaction.shading.n = input.Normal;
    
    Material material;
    material.Init(surface_interaction);

    return float4(material.bsdf.Eval(1.f, 1.f, TransportMode_Radiance), 1.f);
}
