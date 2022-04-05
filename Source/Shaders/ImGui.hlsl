Texture2D fontTexture : register(t0);
SamplerState fontSampler : register(s0);

[[vk::push_constant]]
struct
{
    float2 scale;
    float2 translate;
} push_constants;

struct VSInput
{
    [[vk::location(0)]] float2 Pos : POSITION0;
    [[vk::location(1)]] float2 UV : TEXCOORD0;
    [[vk::location(2)]] float4 Color : COLOR0;
};

struct VSOutput
{
    float4 Pos : SV_POSITION;
    [[vk::location(0)]] float2 UV : TEXCOORD0;
    [[vk::location(1)]] float4 Color : COLOR0;
};

struct PSInput
{
    [[vk::location(0)]] float2 UV : TEXCOORD0;
    [[vk::location(1)]] float4 Color : COLOR0;
};

struct PSOutput
{
    [[vk::location(0)]] float4 Color : SV_Target0;
};

VSOutput VSmain(VSInput input)
{
    VSOutput output;
    output.UV = input.UV;
    output.Color = input.Color;
    output.Pos = float4(input.Pos * push_constants.scale + push_constants.translate, 0.0, 1.0);
    return output;
}

PSOutput PSmain(PSInput input)
{
    PSOutput output;
    
    output.Color = input.Color * fontTexture.Sample(fontSampler, input.UV);
    
    return output;
}