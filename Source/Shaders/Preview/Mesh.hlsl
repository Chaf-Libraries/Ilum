struct VSInput
{
    float3 Position : POSITION0;
    float3 Normal : NORMAL0;
    float3 Tangent : TANGENT0;
    float2 Texcoord0 : TEXCOORD0;
    float2 Texcoord1 : TEXCOORD1;
    uint InstanceID : SV_InstanceID;
};

struct VSOutput
{
    float4 Position : SV_Position;
    float3 Direction : POSITION0;
    float3 Normal : NORMAL0;
};

struct PSInput
{
    float3 Direction : POSITION0;
    float3 Normal : NORMAL0;
};

struct UniformBlock
{
    float4x4 transform;
};

ConstantBuffer<UniformBlock> UniformBuffer;

VSOutput VSmain(VSInput input)
{
    VSOutput output = (VSOutput) 0;
    float4 position = mul(UniformBuffer.transform, float4(input.Position, 1.0f));
    output.Position = position;
    output.Normal = input.Normal;
    output.Direction = normalize(float3(0, 1, 1));
    return output;
}

float4 PSmain(PSInput input) : SV_TARGET
{
    float3 norm = normalize(input.Normal);
    float3 diffuse = max(dot(norm, input.Direction), 0.0);
    return float4(diffuse, 1.f);
}