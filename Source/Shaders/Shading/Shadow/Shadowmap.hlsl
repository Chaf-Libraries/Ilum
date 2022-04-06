#include "../../Common.hlsli"
#include "../../Light.hlsli"

StructuredBuffer<Instance> instances : register(t0);
StructuredBuffer<SpotLight> spot_lights : register(t1);

[[vk::push_constant]]
struct
{
    float4x4 transform;
    uint dynamic;
    uint layer;
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
    float4 Pos : SV_Position;
    uint Layer : SV_RenderTargetArrayIndex;
};

VSOutput main(VSInput input)
{
    VSOutput output;
    float4x4 trans = push_constants.dynamic == 1 ? push_constants.transform : instances[input.InstanceID].transform;
    output.Layer = push_constants.layer;
    output.Pos = mul(spot_lights[push_constants.layer].view_projection, mul(trans, float4(input.Pos, 1.0)));
    return output;
}