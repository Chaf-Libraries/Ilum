#include "../Constants.hlsli"
#include "../Random.hlsli"

#define POSITIVE_X 0
#define NEGATIVE_X 1
#define POSITIVE_Y 2
#define NEGATIVE_Y 3
#define POSITIVE_Z 4
#define NEGATIVE_Z 5

#define LOCAL_SIZE 8

#define SAMPLE_COUNT 4096

RWTexture2DArray<float4> PrefilterMap : register(u0);
TextureCube Skybox : register(t1);
SamplerState SkyboxSampler : register(s1);

[[vk::push_constant]]
struct
{
    uint2 mip_extent;
    float roughness;
} push_constants;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

float3 CalculateDirection(uint face_idx, uint face_x, uint face_y)
{
    float u = 2.0 * (float(face_x) + 0.5) / float(push_constants.mip_extent.x) - 1.0;
    float v = 2.0 * (float(face_y) + 0.5) / float(push_constants.mip_extent.y) - 1.0;
    float x, y, z;

    switch (face_idx)
    {
        case POSITIVE_X:
            x = 1;
            y = -v;
            z = -u;
            break;
        case NEGATIVE_X:
            x = -1;
            y = -v;
            z = u;
            break;
        case POSITIVE_Y:
            x = u;
            y = 1;
            z = v;
            break;
        case NEGATIVE_Y:
            x = u;
            y = -1;
            z = -v;
            break;
        case POSITIVE_Z:
            x = u;
            y = -v;
            z = 1;
            break;
        case NEGATIVE_Z:
            x = -u;
            y = -v;
            z = -1;
            break;
    }
	
    return normalize(float3(x, y, z));
}

float3 GGXImportanceSampling(float2 Xi, float3 N, float roughness)
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

[numthreads(LOCAL_SIZE, LOCAL_SIZE, 1)]
void main(CSParam param)
{
    float3 N = CalculateDirection(param.DispatchThreadID.z, param.DispatchThreadID.x, param.DispatchThreadID.y);
    float3 R = N;
    float3 V = R;

    int2 size;
    Skybox.GetDimensions(size.x, size.y);

    float resolution = float(size.x);

    float3 prefilter_color = float3(0.0, 0.0, 0.0);
    float total_weight = 0.0;

    for (uint i = 0; i < SAMPLE_COUNT; i++)
    {
        float2 Xi = Hammersley(i, SAMPLE_COUNT);
        float3 H = GGXImportanceSampling(Xi, N, push_constants.roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);

        float NoL = max(dot(N, L), 0.0);
        if (NoL > 0.0)
        {
            prefilter_color += Skybox.SampleLevel(SkyboxSampler, L, 0.0).rgb * NoL;
            total_weight += NoL;
        }
    }

    prefilter_color /= total_weight;
    PrefilterMap[int3(param.DispatchThreadID.xyz)] = float4(prefilter_color, 1.0);
}