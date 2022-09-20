struct VSInput
{
    float2 Pos : POSITIONT0;
    float2 UV : TEXCOORD0;
    float4 Color : COLOR0;
};

struct Constant
{
    float2 scale;
    float2 translate;
};

ConstantBuffer<Constant> constant : register(b0);

Texture2D fontTexture : register(t1);
SamplerState fontSampler : register(s2);

struct VSOutput
{
    float4 Pos : SV_Position;
    float2 UV : TEXCOORD0;
    float4 Color : COLOR0;
};

VSOutput VSmain(VSInput input)
{
    VSOutput output = (VSOutput) 0;
    output.UV = input.UV;
    output.Color = input.Color;
    output.Pos = float4(input.Pos * constant.scale + constant.translate, 0.0, 1.0);
#ifdef VULKAN_BACKEND
    output.Pos.y *= -1.f;
#endif
    return output;
}

struct PSInput
{
    float2 UV : TEXCOORD0;
    float4 Color : COLOR0;
};

float4 PSmain(PSInput input) : SV_TARGET
{
    return input.Color * fontTexture.Sample(fontSampler, input.UV);
}
