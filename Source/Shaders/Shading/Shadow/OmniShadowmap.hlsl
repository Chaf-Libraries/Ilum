#include "../../Common.hlsli"

StructuredBuffer<Instance> instances : register(t0);
StructuredBuffer<PointLight> point_lights : register(t1);

[[vk::push_constant]]
struct
{
    float4x4 transform;
    float4x4 view_projection;
    float3 light_pos;
    uint dynamic;
    uint light_id;
    uint face_id;
    float depth_bias;
} push_constants;

struct VSInput
{
    uint InstanceID : SV_InstanceID;
    [[vk::location(0)]] float3 Pos : POSITION0;
    [[vk::location(1)]] float2 UV : TEXCOORD0;
    [[vk::location(2)]] float3 Normal : COLOR0;
};

struct VSOutput
{
    float4 Pos_ : SV_Position;
    uint Layer : SV_RenderTargetArrayIndex;
    float4 Pos : POSITION0;
    float3 LightPos : POSITION1;
    float DepthBias : COLOR0;
};

struct PSInput
{
    float4 Pos : POSITION0;
    float3 LightPos : POSITION1;
    float DepthBias : COLOR0;
};

struct PSOutput
{
    float depth : SV_Depth;
};

VSOutput VSmain(VSInput input)
{
    VSOutput output;
    float4x4 trans = push_constants.dynamic == 1 ? push_constants.transform : instances[input.InstanceID].transform;
    output.Layer = push_constants.light_id * 6 + push_constants.face_id;
    output.Pos_ = mul(push_constants.view_projection, mul(trans, float4(input.Pos, 1.0)));
    
    output.Pos = mul(trans, float4(input.Pos, 1.0));
    output.LightPos = push_constants.light_pos;
    output.DepthBias = push_constants.depth_bias;
    
    return output;
}

PSOutput PSmain(PSInput input)
{
    PSOutput output;
    output.depth = (length(input.Pos.xyz - input.LightPos.xyz) + input.DepthBias);
    return output;
}