#include "Random.hlsli"

#define LOCAL_SIZE 32
#define LUT_SIZE 1024
#define SAMPLE_COUNT 1024

#ifdef Energy
RWTexture2D<float> Emu_Lut : register(u0);
#endif

#ifdef Average
RWTexture2D<float> Eavg_Lut : register(u0);
Texture2D<float> Emu_Lut : register(t1);
#endif

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

float IntegrateBRDF(float3 V, float roughness)
{
    float3 N = float3(0.0, 0.0, 1.0);
    float A = 0.0;

    for (uint i = 0; i < SAMPLE_COUNT; i++)
    {
        float2 Xi = Hammersley(i, SAMPLE_COUNT);
        float3 H = ImportanceSampleGGX(Xi, N, roughness);
        float3 L = normalize(H * 2.0 * dot(V, H) - V);

        float NoV = clamp(dot(N, V), 0.0, 1.0);
        float NoL = clamp(dot(N, L), 0.0, 1.0);
        float NoH = clamp(dot(N, H), 0.0, 1.0);
        float VoH = clamp(dot(V, H), 0.0, 1.0);

        float G = GeometrySmith(roughness, NoV, NoL);
        float w = VoH * G / (NoV * NoH);

        A += w;
    }

    return A / SAMPLE_COUNT;
}

float3 IntegrateEmu(float3 V, float roughness, float NoV, float3 Ei)
{
    float3 Eavg = float3(0.0, 0.0, 0.0);
    float3 N = float3(0.0, 0.0, 1.0);

    for (uint i = 0; i < SAMPLE_COUNT; i++)
    {
        float2 Xi = Hammersley(i, SAMPLE_COUNT);
        float3 H = ImportanceSampleGGX(Xi, N, roughness);
        float3 L = normalize(H * 2.0 * dot(V, H) - V);

        float NoL = clamp(L.z, 0.0, 1.0);

        Eavg += Ei * 2.0 * NoL;
    }

    return Eavg / SAMPLE_COUNT;
}

#ifdef Energy
[numthreads(LOCAL_SIZE, LOCAL_SIZE, 1)]
void main(CSParam param)
{
    float2 uv = float2(param.DispatchThreadID.xy);
    float step = 1.0 / LUT_SIZE;

    float roughness = step * (uv.y + 0.5);
    float NoV = step * (uv.x + 0.5);
    float3 V = float3(sqrt(1.0 - NoV * NoV), 0.0, NoV);

    Emu_Lut[int2(param.DispatchThreadID.xy)] = IntegrateBRDF(V, roughness);
}
#endif

#ifdef Average
[numthreads(LOCAL_SIZE, 1, 1)]
void main(CSParam param)
{
    float step = 1.0 / LUT_SIZE;
    float3 Eavg = float3(0.0, 0.0, 0.0);

    float roughness = step * (float(param.DispatchThreadID.x) + 0.5);

    for (uint i = 0; i < LUT_SIZE; i++)
    {
        float NoV = step * (float(i) + 0.5);
        float3 V = float3(sqrt(1.0 - NoV * NoV), 0.0, NoV);
        float c = Emu_Lut.Load(int3(i, param.DispatchThreadID.x, 0));
        float3 Ei = float3(c, c, c);
        Eavg += IntegrateEmu(V, roughness, NoV, Ei) * step;
    }
    
    for (i = 0; i < LUT_SIZE; i++)
    {
        Eavg_Lut[int2(i, param.DispatchThreadID.x)] = Eavg.r;
    }
}
#endif