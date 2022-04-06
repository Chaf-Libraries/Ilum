#include "../../Common.hlsli"
#include "../../Light.hlsli"

StructuredBuffer<Instance> instances : register(t0);
StructuredBuffer<DirectionalLight> directionl_lights : register(t1);

[[vk::push_constant]]
struct
{
    float4x4 transform;
    uint dynamic;
    uint light_id;
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
    uint InstanceID : POSITION0;
};

struct GSInput
{
    float4 Pos : SV_Position;
    uint InstanceID : POSITION0;
};

struct GSOutput
{
    float4 Pos : SV_Position;
    uint Layer : SV_RenderTargetArrayIndex;
};

VSOutput VSmain(VSInput input)
{
    VSOutput output;
    output.InstanceID = input.InstanceID;
    output.Pos = float4(input.Pos, 1.0);
    return output;
}

[maxvertexcount(3)]
[instance(4)]
void GSmain(triangle GSInput input[4], uint InvocationID : SV_GSInstanceID, inout TriangleStream<GSOutput> outStream)
{
    float4x4 transform = push_constants.dynamic == 1 ? push_constants.transform : instances[input[0].InstanceID].transform;
    
    for (uint i = 0; i < 4; i++)
    {
        GSOutput output;
        output.Pos = mul(directionl_lights[push_constants.light_id].view_projection[InvocationID], mul(transform, input[i].Pos));
        output.Layer = push_constants.light_id * 4 + InvocationID;
        outStream.Append(output);
    }
    outStream.RestartStrip();

}