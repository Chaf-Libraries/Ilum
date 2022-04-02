#include "../Random.hlsli"

#define LOCAL_SIZE 32
#define LUT_SIZE 1024
#define SAMPLE_COUNT 1024

RWTexture2D<float> Emu_Lut : register(u0);
RWTexture2D<float> Eavg_Lut : register(u1);

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
       
        Emu_Lut[int2(i / LUT_SIZE, param.DispatchThreadID.x / LUT_SIZE)];
        
        float c = Emu_Lut[int2(i / LUT_SIZE, param.DispatchThreadID.x / LUT_SIZE)].r;
        float3 Ei = float3(c, c, c);
        Eavg += IntegrateEmu(V, roughness, NoV, Ei) * step;
    }
    
    for (i = 0; i < LUT_SIZE; i++)
    {
        Eavg_Lut[int2(i, param.DispatchThreadID.x)] = Eavg.r;
    }
}