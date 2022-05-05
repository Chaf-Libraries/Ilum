#include "../Random.hlsli"

#define LOCAL_SIZE 32
#define SAMPLE_COUNT 4096
#define LUT_SIZE 516

RWTexture2D<float2> BRDFPreIntegrate : register(u0);

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

float3 ImportanceSampleGGX(float2 Xi, float3 N, float roughness)
{
    float alpha = roughness * roughness;

    float theta = atan(alpha * sqrt(Xi.x) / sqrt(1 - Xi.x));
    float phi = 2.0 * PI * Xi.y;

    float3 H = float3(
        sin(theta) * cos(phi),
        sin(theta) * sin(phi),
        cos(theta)
    );

    float3 Up = N.z > 0.99 ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0);
    float3 T = normalize(cross(N, Up));
    float3 B = normalize(cross(N, T));

    return T * H.x + B * H.y + N * H.z;
}

float conv(in float kernel[9], in float data[9], in float denom, in float offset)
{
    float res = 0.0;
    for (int i = 0; i < 9; ++i)
    {
        res += kernel[i] * data[i];
    }
    return saturate(res / denom + offset);
}

float GeometrySchlickGGX(float NoV, float roughness)
{
    float alpha = roughness;
    float k = alpha * alpha / 2.0;

    return NoV / (NoV * (1.0 - k) + k);
}

float GeometrySmith(float roughness, float NoV, float NoL)
{
    return GeometrySchlickGGX(NoV, roughness) * GeometrySchlickGGX(NoL, roughness);
}

float2 IntegrateBRDF(float NoV, float roughness)
{
    float3 V;
    V.x = sqrt(1.0 - NoV * NoV);
    V.y = 0.0;
    V.z = NoV;

    float A = 0.0;
    float B = 0.0;

    float3 N = float3(0.0, 0.0, 1.0);

    for (uint i = 0; i < SAMPLE_COUNT; i++)
    {
        float2 Xi = Hammersley(i, SAMPLE_COUNT);
        float3 H = ImportanceSampleGGX(Xi, N, roughness);
        float3 L = normalize(H * 2.0 * dot(V, H) - V);

        float NoL = clamp(dot(N, L), 0.0, 1.0);
        float NoH = clamp(dot(N, H), 0.0, 1.0);
        float VoH = clamp(dot(V, H), 0.0, 1.0);

        if (NoL > 0.0)
        {
            float G = GeometrySmith(roughness, NoV, NoL);
            float w = VoH * G / (NoV * NoH);
            float Fc = pow(1.0 - VoH, 5.0);

            A += (1.0 - Fc) * w;
            B += Fc * w;
        }
    }

    return float2(A, B) / float(SAMPLE_COUNT);
}

[numthreads(LOCAL_SIZE, LOCAL_SIZE, 1)]
void main(CSParam param)
{
    float2 tex_coord = float2(float2(param.DispatchThreadID.xy) + float2(0.5, 0.5)) / float(LUT_SIZE);
    float2 brdf = IntegrateBRDF(tex_coord.x, tex_coord.y);

    BRDFPreIntegrate[int2(param.DispatchThreadID.xy)] = brdf;
}
