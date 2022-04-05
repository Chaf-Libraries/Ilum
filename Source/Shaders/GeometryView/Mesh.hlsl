#include "../Common.hlsli"

ConstantBuffer<Camera> camera : register(b0);
Texture2D textureArray[] : register(t1);
SamplerState texSampler : register(s1);

[[vk::push_constant]]
struct
{
    float4x4 transform;
    uint texture_id;
    uint parameterization;
} push_constants;

struct VSInput
{
    [[vk::location(0)]] float3 Pos : POSITION0;
    [[vk::location(1)]] float2 UV : TEXCOORD0;
    [[vk::location(2)]] float3 Normal : COLOR0;
};

struct VSOutput
{
    float4 Pos : SV_POSITION;
    [[vk::location(0)]] float2 UV : TEXCOORD0;
    [[vk::location(1)]] uint TexID : POSITION0;
};

struct PSInput
{
    [[vk::location(0)]] float2 UV : TEXCOORD0;
    [[vk::location(1)]] uint TexID : POSITION0;
};

struct PSOutput
{
    [[vk::location(0)]] float4 Color : SV_Target0;
};

VSOutput VSmain(VSInput input)
{
    VSOutput output;
    if (push_constants.parameterization == 1)
    {
        output.Pos = float4(2.0 * input.UV - 1.0, 0.0, 1.0);
    }
    else
    {
        output.Pos = mul(camera.view_projection, mul(push_constants.transform, float4(input.Pos, 1.0)));
    }
    
    output.UV = input.UV;
    output.TexID = push_constants.texture_id;
    return output;
}

PSOutput PSmain(PSInput input)
{
    PSOutput output;
    if (input.TexID < 1024)
    {
        output.Color = float4(textureArray[NonUniformResourceIndex(input.TexID)].SampleLevel(texSampler, input.UV, 0.0).rgb, 1.0);
    }
    else
    {
        output.Color = float4(input.UV, 0.0, 1.0);
    }
    return output;
}