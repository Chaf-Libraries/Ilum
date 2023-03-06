//#define SHADING
//#define DRAW_NORMAL
//#define DRAW_UV
//#define DRAW_TEXTURE
//#define WIREFRAME

struct VSInput
{
    float3 Position : POSITIONT0;
#if !defined(DRAW_UV) && !defined(WIREFRAME)
    float3 Normal : NORMAL0;
#endif
#if defined(DRAW_UV) || defined(DRAW_TEXTURE)
    float2 UV : TEXCOORD0;
#endif
};

struct UniformBlock
{
    float4x4 transform;
    float3 color;
    float3 direction;
};

Texture2D UVTexture;
SamplerState UVSampler;
ConstantBuffer<UniformBlock> UniformBuffer;

struct VSOutput
{
    float4 Position : SV_Position;
#if !defined(DRAW_UV) && !defined(WIREFRAME)
    float3 Normal : NORMAL0;
#endif
#if defined(DRAW_UV) || defined(DRAW_TEXTURE)
    float2 UV : TEXCOORD0;
#endif
};

VSOutput VSmain(VSInput input)
{
    VSOutput output = (VSOutput) 0;
#if defined(DRAW_UV) || defined(DRAW_TEXTURE)
    output.UV = input.UV;
#endif
    output.Position = mul(UniformBuffer.transform, float4(input.Position, 1.0f));
#if !defined(DRAW_UV) && !defined(WIREFRAME)
    output.Normal = input.Normal;
#endif
    return output;
}

struct PSInput
{
#if !defined(DRAW_UV) && !defined(WIREFRAME)
    float3 Normal : NORMAL0;
#endif
#if defined(DRAW_UV) || defined(DRAW_TEXTURE)
    float2 UV : TEXCOORD0;
#endif
};

float4 PSmain(PSInput input) : SV_TARGET
{
#ifdef SHADING
    float3 ambient = 0.3f;
    float3 norm = normalize(input.Normal);

    float3 diffuse = max(dot(norm, UniformBuffer.direction), 0.0);
    
    float3 reflect_dir = reflect(-UniformBuffer.direction, norm);
    float specular = pow(max(dot(UniformBuffer.direction, reflect_dir), 0.0), 3.f);
    return float4((ambient + diffuse + specular) * UniformBuffer.color, 1.0f);
#endif
    
#ifdef DRAW_NORMAL
    return float4(input.Normal, 1.0f);
#endif
    
#ifdef DRAW_UV
    return float4(input.UV, 0.f, 1.0f);
#endif
    
#ifdef DRAW_TEXTURE
    float3 ambient = 0.3f;
    float3 norm = normalize(input.Normal);
    float3 texture_color = UVTexture.Sample(UVSampler, input.UV).rgb;

    float3 diffuse = max(dot(norm, UniformBuffer.direction), 0.0);
    
    float3 reflect_dir = reflect(-UniformBuffer.direction, norm);
    float specular = pow(max(dot(UniformBuffer.direction, reflect_dir), 0.0), 3.f);
    return float4((ambient + diffuse + specular) * texture_color, 1.0f);
#endif
    
#ifdef WIREFRAME
    return float4(0.f, 0.f, 0.f, 1.f);
#endif
}
