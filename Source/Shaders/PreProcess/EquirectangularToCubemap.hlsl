static const float PI = 3.14159265358979323846f;

struct UniformBlock
{
    float4x4 inv_view_projection[6];
};

Texture2D InputTexture;
SamplerState TexSampler;
ConstantBuffer<UniformBlock> UniformBuffer;

struct VSInput
{
    uint VertexID : SV_VertexID;
    uint InstanceID : SV_InstanceID;
};

struct VSOutput
{
    float3 Pos : POSITION0;
    float2 UV : TEXCOORD0;
    float4 Position : SV_POSITION;
    uint Layer : SV_RenderTargetArrayIndex;
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
    output.Position = float4(output.UV * 2.0 - 1.0, 1.0, 1.0);
    output.Pos = mul(UniformBuffer.inv_view_projection[input.InstanceID], output.Position).xyz;
    output.Layer = input.InstanceID;

    return output;
}

FSOutput PSmain(FSInput input)
{
    FSOutput output;
    
    float2 UV = SampleSphericalMap(normalize(input.Pos));
    output.Color = float4(InputTexture.Sample(TexSampler, UV).rgb, 1.0);
    // output.Color = float4(UV, 0.0, 1.0);
    
    return output;
}