#include "../../Common.hlsli"
#include "../../Material.hlsli"
#include "../../Math.hlsli"

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
};

struct VSOutput
{
    float4 Pos_ : SV_POSITION;
    float4 Pos : POSITION0;
    float2 UV : TEXCOORD0;
    float3 Normal : COLOR0;
    uint InstanceID : COLOR1;
    uint EntityID : COLOR2;
    float4 ScreenSpacePos : POSITION1;
    float4 LastScreenSpacePos : POSITION2;
};

struct PSInput
{
    float4 Pos : POSITION0;
    float2 UV : TEXCOORD0;
    float3 Normal : COLOR0;
    uint InstanceID : COLOR1;
    uint EntityID : COLOR2;
    float4 ScreenSpacePos : POSITION1;
    float4 LastScreenSpacePos : POSITION2;
};

struct PSOutput
{
    float4 GBuffer0 : SV_Target0; // GBuffer 0: RGB - Albedo, A - Anisotropic
    float4 GBuffer1 : SV_Target1; // GBuffer 1: RGB - Normal, A - LinearDepth
    float4 GBuffer2 : SV_Target2; // GBuffer 2: R - Metallic, G - Roughness, B - Subsurface, A - EntityID
    float4 GBuffer3 : SV_Target3; // GBuffer 3: R - Sheen, G - Sheen Tint, B - Clearcoat, A - Clearcoat Gloss
    float4 GBuffer4 : SV_Target4; // GBuffer 4: RG - Velocity, B - Specular, A - Specular Tint
    float4 GBuffer5 : SV_Target5; // GBuffer 5: RGB - Emissive, A - Material Type
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
    output.InstanceID = input.InstanceID;
    output.EntityID = push_constants.dynamic == 1 ? push_constants.entity_id : instances[input.InstanceID].entity_id;
      
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
    
    // GBuffer 0: RGB - Albedo, A - Anisotropic
    if (materials[input.InstanceID].textures[TEXTURE_BASE_COLOR] < MAX_TEXTURE_ARRAY_SIZE)
    {
        output.GBuffer0.rgb = textureArray[NonUniformResourceIndex(materials[input.InstanceID].textures[TEXTURE_BASE_COLOR])].Sample(texSampler, input.UV).rgb
                                        * materials[input.InstanceID].base_color.rgb;
    }
    else
    {
        output.GBuffer0.rgb = materials[input.InstanceID].base_color.rgb;
    }
    output.GBuffer0.a = materials[input.InstanceID].anisotropic;
    
    // GBuffer 1: RGB - Normal, A - LinearDepth
    if (materials[input.InstanceID].textures[TEXTURE_NORMAL] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float3 normal = textureArray[NonUniformResourceIndex(materials[input.InstanceID].textures[TEXTURE_NORMAL])].Sample(texSampler, input.UV).rgb;
        output.GBuffer1.rgb = normalize(mul(normalize(normal * 2.0 - 1.0), TBN));
    }
    else
    {
        output.GBuffer1.rgb = N;
    }
    output.GBuffer1.a = input.Pos.w;
    
    // GBuffer 2: R - Metallic, G - Roughness, B - Subsurface, A - EntityID
    if (materials[input.InstanceID].textures[TEXTURE_METALLIC] < MAX_TEXTURE_ARRAY_SIZE)
    {
        output.GBuffer2.r = textureArray[NonUniformResourceIndex(materials[input.InstanceID].textures[TEXTURE_METALLIC])].Sample(texSampler, input.UV).r
                                    * materials[input.InstanceID].metallic;
    }
    else
    {
        output.GBuffer2.r = materials[input.InstanceID].metallic;
    }
    
    if (materials[input.InstanceID].textures[TEXTURE_ROUGHNESS] < MAX_TEXTURE_ARRAY_SIZE)
    {
        output.GBuffer2.g = textureArray[NonUniformResourceIndex(materials[input.InstanceID].textures[TEXTURE_ROUGHNESS])].Sample(texSampler, input.UV).g
                                    * materials[input.InstanceID].roughness;
    }
    else
    {
        output.GBuffer2.g = materials[input.InstanceID].roughness;
    }
    output.GBuffer2.b = materials[input.InstanceID].subsurface;
    output.GBuffer2.a = input.EntityID;
    
    // GBuffer 3: R - Sheen, G - Sheen Tint, B - Clearcoat, A - Clearcoat Gloss
    output.GBuffer3.r = materials[input.InstanceID].sheen;
    output.GBuffer3.g = materials[input.InstanceID].sheen_tint;
    output.GBuffer3.b = materials[input.InstanceID].clearcoat;
    output.GBuffer3.a = materials[input.InstanceID].clearcoat_gloss;
    
    // GBuffer 4: RG - Velocity, B - Specular, A - Specular Tint
    output.GBuffer4.rg = ComputeMotionVector(input.LastScreenSpacePos, input.ScreenSpacePos);
    output.GBuffer4.b = materials[input.InstanceID].specular;
    output.GBuffer4.a = materials[input.InstanceID].specular_tint;
    
    // GBuffer 5: RGB - Emissive, A - Material Type
    if (materials[input.InstanceID].textures[TEXTURE_EMISSIVE] < MAX_TEXTURE_ARRAY_SIZE)
    {
        output.GBuffer5.rgb = textureArray[NonUniformResourceIndex(materials[input.InstanceID].textures[TEXTURE_EMISSIVE])].Sample(texSampler, input.UV).rgb
                                    * materials[input.InstanceID].emissive_color * materials[input.InstanceID].emissive_intensity;
    }
    else
    {
        output.GBuffer5.rgb = materials[input.InstanceID].emissive_color * materials[input.InstanceID].emissive_intensity;
    }
    output.GBuffer5.a = materials[input.InstanceID].material_type;
    
    return output;
}