#include "../Common.hlsli"
#include "../Material.hlsli"
#include "../Math.hlsli"

ConstantBuffer<Camera> camera : register(b0);
Texture2D textureArray[] : register(t1);
SamplerState texSampler : register(s1);
StructuredBuffer<Instance> instances : register(t2);
StructuredBuffer<Meshlet> meshlets : register(t3);
StructuredBuffer<MaterialData> materials : register(t4);

[[vk::push_constant]]
struct
{
    float4x4 transform;
    uint dynamic;
    uint entity_id;
} push_constants;

struct VSInput
{
    uint InstanceID : SV_InstanceID;
    [[vk::location(0)]] float3 Pos : POSITION0;
    [[vk::location(1)]] float2 UV : TEXCOORD0;
    [[vk::location(2)]] float3 Normal : COLOR0;
    [[vk::location(3)]] float3 Tangent : COLOR1;
    [[vk::location(4)]] float3 Bitangent : COLOR2;
};

struct VSOutput
{
    float4 Pos_ : SV_POSITION;
    float4 Pos : POSITION0;
    float2 UV : TEXCOORD0;
    float3 Normal : COLOR0;
    float3 Tangent : COLOR1;
    float3 Bitangent : COLOR2;
    uint2 Data : COLOR3; // x - InstanceID, y - Entity ID
    float4 ScreenSpacePos : POSITION1;
    float4 LastScreenSpacePos : POSITION2;
};

struct PSInput
{
    float4 Pos : POSITION0;
    float2 UV : TEXCOORD0;
    float3 Normal : COLOR0;
    float3 Tangent : COLOR1;
    float3 Bitangent : COLOR2;
    uint2 Data : COLOR3; // x - InstanceID, y - Entity ID
    float4 ScreenSpacePos : POSITION1;
    float4 LastScreenSpacePos : POSITION2;
};

struct PSOutput
{
    float4 GBuffer0 : SV_Target0; // RGB - Normal, A - Linear Depth
    float4 GBuffer1 : SV_Target1; // RG - Velocity, B - Instance ID, A - Entity ID, 
    float2 GBuffer2 : SV_Target2; // RG - UV
};

float2 ComputeMotionVector(float4 prev_pos, float4 current_pos)
{
    // Clip space -> NDC
    float2 current = current_pos.xy / current_pos.w;
    float2 prev = prev_pos.xy / prev_pos.w;

    current = current * 0.5 + 0.5;
    prev = prev * 0.5 + 0.5;

    current.y = 1 - current.y;
    prev.y = 1 - prev.y;

    return current - prev;
}

VSOutput VSmain(VSInput input)
{
    VSOutput output;
    output.UV = input.UV;
    output.Data.x = input.InstanceID;
    output.Data.y = push_constants.dynamic == 1 ? push_constants.entity_id : instances[input.InstanceID].entity_id;
      
    float height;
    if (materials[input.InstanceID].textures[TEXTURE_DISPLACEMENT] < MAX_TEXTURE_ARRAY_SIZE)
    {
        height = max(0.0, textureArray[NonUniformResourceIndex(materials[input.InstanceID].textures[TEXTURE_DISPLACEMENT])].SampleLevel(texSampler, input.UV, 0.0).r) * materials[input.InstanceID].displacement;
    }
    else
    {
        height = 0.0;
    }
       
    float4x4 transform = push_constants.dynamic == 1 ? push_constants.transform : instances[input.InstanceID].transform;
    
    output.Normal = mul((float3x3) transform, input.Normal);
    output.Tangent = input.Tangent;
    
    output.Pos = mul(transform, float4(input.Pos, 1.0));
    output.Pos.xyz += normalize(output.Normal) * height;
    output.Pos_ = mul(camera.view_projection, output.Pos);
    
    output.ScreenSpacePos = output.Pos_;
    output.LastScreenSpacePos =
        push_constants.dynamic == 1 ?
        float4(0.0, 0.0, 0.0, 0.0) :
        mul(camera.last_view_projection, mul(instances[input.InstanceID].last_transform, float4(input.Pos, 1.0)));
    
    return output;
}

PSOutput PSmain(PSInput input)
{
    PSOutput output;
    
    float3 N = normalize(input.Normal);
    float3 T, B;
    CreateCoordinateSystem(N, T, B);
    
    float3x3 TBN = float3x3(T, B, N);
    
    if (materials[input.Data.x].textures[TEXTURE_NORMAL] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float3 normal = textureArray[NonUniformResourceIndex(materials[input.Data.x].textures[TEXTURE_NORMAL])].Sample(texSampler, input.UV).rgb;
        output.GBuffer0.rgb = normalize(mul(normalize(normal * 2.0 - 1.0), TBN));
    }
    else
    {
        output.GBuffer0.rgb = N;
    }
    output.GBuffer0.a = input.Pos.w;
    
    output.GBuffer1.rg = ComputeMotionVector(input.LastScreenSpacePos, input.ScreenSpacePos);
    output.GBuffer1.b = input.Data.x;
    output.GBuffer1.a = input.Data.y;
    
    output.GBuffer2.rg = input.UV;
    
    return output;
}