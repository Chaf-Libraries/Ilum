#include "Constants.hlsli"

Texture2D tex : register(t0);
SamplerState texSampler : register(s1);

[[vk::push_constant]]
struct
{
    float4x4 inverse_view_projection;
} push_constants;

struct VSInput
{
    uint VertexID : SV_VertexID;
};

struct VSOutput
{
    float4 Vertex_Position : SV_POSITION;
    float3 Pos : POSITION0;
    float2 UV : TEXCOORD0;
};

struct FSInput
{
    float3 Pos : POSITION0;
    float2 UV : TEXCOORD0;
};

struct FSOutput
{
    float4 Color : SV_Target0;
};

float2 SampleSphericalMap(float3 v)
{
    float2 uv = float2(atan2(v.x, v.z), asin(v.y));
    uv.x /= 2 * PI;
    uv.y /= PI;
    uv += 0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}

VSOutput VSmain(VSInput input)
{
    VSOutput output = (VSOutput) 0;
    
    output.UV = float2((input.VertexID << 1) & 2, input.VertexID & 2);
    output.Vertex_Position = float4(output.UV * 2.0 - 1.0, 1.0, 1.0);
    output.Pos = mul(push_constants.inverse_view_projection, output.Vertex_Position).xyz;

    return output;
}

FSOutput PSmain(FSInput input)
{
    FSOutput output;
    
    float2 uv = SampleSphericalMap(normalize(input.Pos));
    output.Color = float4(tex.Sample(texSampler, uv).rgb, 1.0);
    return output;
}