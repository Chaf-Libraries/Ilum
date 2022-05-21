#include "Random.hlsli"
#include "BxDF.hlsli"

#define SAMPLE_COUNT 1024

RWTexture2D<float2> LUT_GGX : register(u0);
RWTexture2D<float> LUT_Charlie : register(u1);

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

struct MicrofacetDistributionSample
{
    float pdf;
    float cosTheta;
    float sinTheta;
    float phi;
};

MicrofacetDistributionSample SampleGGX(float2 xi, float roughness)
{
    MicrofacetDistributionSample ggx;
    
    float alpha = roughness * roughness;
    ggx.cosTheta = saturate(sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y)));
    ggx.sinTheta = sqrt(1.0 - ggx.cosTheta * ggx.cosTheta);
    ggx.phi = 2.0 * PI * xi.x;
    ggx.pdf = D_GGX(ggx.cosTheta, alpha) / 4.0;
    return ggx;
}

MicrofacetDistributionSample SampleCharlie(float2 xi, float roughness)
{
    MicrofacetDistributionSample charlie;
    
    float alpha = roughness * roughness;
    charlie.sinTheta = pow(xi.y, alpha / (2.0 * alpha + 1.0));
    charlie.cosTheta = sqrt(1.0 - charlie.sinTheta * charlie.sinTheta);
    charlie.phi = 2.0 * PI * xi.x;
    charlie.pdf = D_Charlie(alpha, charlie.cosTheta) / 4.0;
    return charlie;
}

float2 GenerateLUTGGX(float NoV, float roughness)
{
    float3 V = float3(sqrt(1.0 - NoV * NoV), 0.0, NoV);
    float3 N = float3(0.0, 0.0, 1.0);
    float A = 0.0, B = 0.0;
    for (uint i = 0; i < SAMPLE_COUNT; i++)
    {
        float2 xi = Hammersley(i, SAMPLE_COUNT);
        MicrofacetDistributionSample ggx = SampleGGX(xi, roughness);
        float3 local_dir = normalize(float3(
            ggx.sinTheta * cos(ggx.phi),
            ggx.sinTheta * sin(ggx.phi),
            ggx.cosTheta));
        float3 Tangent, Bitangent;
        float3x3 TBN = CreateCoordinateSystem(N, Tangent, Bitangent);
        float3 H = mul(TBN, local_dir);
        float3 L = normalize(reflect(-V, H));
        
        float NoL = saturate(L.z);
        float NoH = saturate(H.z);
        float VoH = saturate(dot(V, H));
        
        if (NoL > 0.0)
        {
            float V_pdf = V_GGX(NoL, NoV, roughness * roughness) * VoH * NoL / NoH;
            float Fc = pow(1.0 - VoH, 5.0);
            A += (1.0 - Fc) * V_pdf;
            B += Fc * V_pdf;
        }
    }
    return float2(4.0 * A, 4.0 * B) / float(SAMPLE_COUNT);
}

float GenerateCharlieLUT(float NoV, float roughness)
{
    float3 V = float3(sqrt(1.0 - NoV * NoV), 0.0, NoV);
    float3 N = float3(0.0, 0.0, 1.0);
    float A = 0.0;
    for (uint i = 0; i < SAMPLE_COUNT; i++)
    {
        float2 xi = Hammersley(i, SAMPLE_COUNT);
        MicrofacetDistributionSample charlie = SampleCharlie(xi, roughness);
        float3 local_dir = normalize(float3(
            charlie.sinTheta * cos(charlie.phi),
            charlie.sinTheta * sin(charlie.phi),
            charlie.cosTheta));
        float3 Tangent, Bitangent;
        float3x3 TBN = CreateCoordinateSystem(N, Tangent, Bitangent);
        float3 H = mul(TBN, local_dir);
        float3 L = normalize(reflect(-V, H));
        
        float NoL = saturate(L.z);
        float NoH = saturate(H.z);
        float VoH = saturate(dot(V, H));
        if (NoL > 0.0)
        {
            float sheen_distribution = D_Charlie(roughness, NoH);
            float sheen_visibility = V_Sheen(NoL, NoV, roughness);
            A += sheen_visibility * sheen_distribution * NoL * VoH;
        }
    }
    return 4.0 * 2.0 * PI * A / float(SAMPLE_COUNT);
}

[numthreads(32, 32, 1)]
void main(CSParam param)
{
    // GGX LUT
    {
        uint2 lut_size;
        LUT_GGX.GetDimensions(lut_size.x, lut_size.y);
        if (lut_size.x > param.DispatchThreadID.x || lut_size.y > param.DispatchThreadID.y)
        {
            float2 tex_coord = float2(float2(param.DispatchThreadID.xy) + float2(0.5, 0.5)) / float2(lut_size);
            LUT_GGX[param.DispatchThreadID.xy] = GenerateLUTGGX(float(tex_coord.x), float(tex_coord.y));
        }
    }
    
    // Charlie LUT
    {
        uint2 lut_size;
        LUT_Charlie.GetDimensions(lut_size.x, lut_size.y);
        if (lut_size.x > param.DispatchThreadID.x || lut_size.y > param.DispatchThreadID.y)
        {
            float2 tex_coord = float2(float2(param.DispatchThreadID.xy) + float2(0.5, 0.5)) / float2(lut_size);
            LUT_Charlie[param.DispatchThreadID.xy] = GenerateCharlieLUT(float(tex_coord.x), float(tex_coord.y));
        }
    }

}