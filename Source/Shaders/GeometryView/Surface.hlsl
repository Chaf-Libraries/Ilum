#include "../Common.hlsli"

ConstantBuffer<Camera> camera : register(b0);

[[vk::push_constant]]
struct
{
    float4x4 transform;
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
    [[vk::location(0)]] float4 Color : COLOR0;
};

struct PSInput
{
    [[vk::location(0)]] float4 Color : COLOR0;
};

struct PSOutput
{
    [[vk::location(0)]] float4 Color : SV_Target0;
};

VSOutput VSmain(VSInput input)
{
    VSOutput output;
    output.Pos = mul(camera.view_projection, mul(push_constants.transform, float4(input.Pos, 1.0)));
    output.Color = float4(input.UV, 0.0, 1.0);
    return output;
}

PSOutput PSmain(PSInput input)
{
    PSOutput output;
    output.Color = input.Color;
    return output;
}