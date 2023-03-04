#include "../Common.hlsli"
#include "../SphericalHarmonic.hlsli"

#define POSITIVE_X 0
#define NEGATIVE_X 1
#define POSITIVE_Y 2
#define NEGATIVE_Y 3
#define POSITIVE_Z 4
#define NEGATIVE_Z 5

#define LOCAL_SIZE 8

#define SH_INTERMEDIATE_SIZE 128 / 8
#define CUBEMAP_FACE_NUM 6
#define PREFILTER_MAP_SIZE 256
#define PREFILTER_MIP_LEVELS 5
#define SAMPLE_COUNT 1024

struct Config
{
    uint2 extent;
};

ConstantBuffer<Config> ConfigBuffer;
RWTexture2DArray<float4> SHIntermediate;
TextureCube Skybox;
SamplerState SkyboxSampler;

groupshared SH9Color projection_shared_sh_coeffs[LOCAL_SIZE][LOCAL_SIZE];
groupshared float projection_shared_weights[LOCAL_SIZE][LOCAL_SIZE];

[numthreads(LOCAL_SIZE, LOCAL_SIZE, 1)]
void CubemapSHProjection(CSParam param)
{
    for (uint i = 0; i < 9; i++)
    {
        projection_shared_sh_coeffs[param.GroupThreadID.x][param.GroupThreadID.y].weights[i] = float3(0.0, 0.0, 0.0);
    }

    GroupMemoryBarrier();
    
    uint2 extent = { 128, 128 };

    SH9 basis;

    float3 dir = CalculateCubemapDirection(param.DispatchThreadID.z, param.DispatchThreadID.x, param.DispatchThreadID.y, extent.x, extent.y);
    float solid_angle = CalculateSolidAngle(param.DispatchThreadID.x, param.DispatchThreadID.y, extent.x, extent.y);
    float3 texel = Skybox.SampleLevel(SkyboxSampler, dir, 0.0).rgb;

    basis = ProjectSH9(dir);

    projection_shared_weights[param.GroupThreadID.x][param.GroupThreadID.y] = solid_angle;

    for (uint i = 0; i < 9; i++)
    {
        projection_shared_sh_coeffs[param.GroupThreadID.x][param.GroupThreadID.y].weights[i] += texel * basis.weights[i] * solid_angle;
    }

    GroupMemoryBarrier();

	// Add up all coefficients and weights along the X axis
    if (param.GroupThreadID.x == 0)
    {
        for (int i = 1; i < LOCAL_SIZE; i++)
        {
            projection_shared_weights[0][param.GroupThreadID.y] += projection_shared_weights[i][param.GroupThreadID.y];
            for (int j = 0; j < 9; j++)
            {
                projection_shared_sh_coeffs[0][param.GroupThreadID.y].weights[j] += projection_shared_sh_coeffs[i][param.GroupThreadID.y].weights[j];
            }
        }
    }

    GroupMemoryBarrier();

	// Add up all coefficients and weights along the Y axis
    if (param.GroupThreadID.x == 0 && param.GroupThreadID.y == 0)
    {
        for (int i = 1; i < LOCAL_SIZE; i++)
        {
            projection_shared_weights[0][0] += projection_shared_weights[0][i];
            for (int j = 0; j < 9; j++)
            {
                projection_shared_sh_coeffs[0][0].weights[j] += projection_shared_sh_coeffs[0][i].weights[j];
            }
        }

        for (int j = 0; j < 9; j++)
        {
            int3 p = int3(extent.x / LOCAL_SIZE * j + param.DispatchThreadID.x / LOCAL_SIZE, param.DispatchThreadID.y / LOCAL_SIZE, param.DispatchThreadID.z);
            
            SHIntermediate[p] = float4(projection_shared_sh_coeffs[0][0].weights[j], projection_shared_weights[0][0]);
        }
    }
}

RWTexture2D<float4> IrradianceSH;

groupshared float3 add_shared_sh_coeffs[SH_INTERMEDIATE_SIZE][CUBEMAP_FACE_NUM];
groupshared float add_shared_weights[SH_INTERMEDIATE_SIZE][CUBEMAP_FACE_NUM];

[numthreads(1, SH_INTERMEDIATE_SIZE, CUBEMAP_FACE_NUM)]
void CubemapSHAdd(CSParam param)
{
    add_shared_sh_coeffs[param.DispatchThreadID.y][param.DispatchThreadID.z] = float3(0.0, 0.0, 0.0);
    add_shared_weights[param.DispatchThreadID.y][param.DispatchThreadID.z] = 0.0;

    GroupMemoryBarrier();

    for (uint i = 0; i < SH_INTERMEDIATE_SIZE; i++)
    {
        uint3 p = uint3(param.DispatchThreadID.x * SH_INTERMEDIATE_SIZE + i, param.DispatchThreadID.y, param.DispatchThreadID.z);
        float4 val = SHIntermediate[p];
        add_shared_sh_coeffs[param.DispatchThreadID.y][param.DispatchThreadID.z] += val.rgb;
        add_shared_weights[param.DispatchThreadID.y][param.DispatchThreadID.z] += val.a;
    }

    GroupMemoryBarrier();

    if (param.DispatchThreadID.z == 0)
    {
        for (uint i = 1; i < CUBEMAP_FACE_NUM; i++)
        {
            add_shared_sh_coeffs[param.DispatchThreadID.y][0] += add_shared_sh_coeffs[param.DispatchThreadID.y][i];
            add_shared_weights[param.DispatchThreadID.y][0] += add_shared_weights[param.DispatchThreadID.y][i];
        }
    }

    GroupMemoryBarrier();

    if (param.DispatchThreadID.y == 0 && param.DispatchThreadID.z == 0)
    {
        for (uint i = 0; i < SH_INTERMEDIATE_SIZE; i++)
        {
            add_shared_sh_coeffs[0][0] += add_shared_sh_coeffs[i][0];
            add_shared_weights[0][0] += add_shared_weights[i][0];
        }

        float scale = (4.0 * PI) / add_shared_weights[0][0];

        IrradianceSH[int2(param.DispatchThreadID.x, 0)] = float4(add_shared_sh_coeffs[0][0] * scale, add_shared_weights[0][0]);
    }
}

RWTexture2DArray<float4> PrefilterMap;

float3 CalculateDirection(uint face_idx, uint face_x, uint face_y, uint2 extent)
{
    float u = 2.0 * (float(face_x) + 0.5) / float(extent.x) - 1.0;
    float v = 2.0 * (float(face_y) + 0.5) / float(extent.y) - 1.0;
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

float GGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NoH = max(dot(N, H), 0.f);
    float NoH2 = NoH * NoH;
    float nom = a2;
    float denom = (NoH2 * (a2 - 1.f) + 1.f);
    denom = PI * denom * denom;
    return nom / denom;
}

float3 GGXImportanceSampling(float2 Xi, float3 N, float roughness)
{
    float a = roughness * roughness;

    float phi = 2.0 * PI * Xi.x;
    float cos_theta = sqrt((1.f - Xi.y) / (1.f + (a * a - 1.f) * Xi.y));
    float sin_theta = sqrt(1.f - cos_theta * cos_theta);

    float3 H = float3(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta);

    float3 Up = N.z > 0.999 ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0);
    float3 T = normalize(cross(N, Up));
    float3 B = normalize(cross(N, T));

    return T * H.x + B * H.y + N * H.z;
}

float RadicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

float2 Hammersley(uint i, uint N)
{
    return float2(float(i) / float(N), RadicalInverse_VdC(i));
}

[numthreads(LOCAL_SIZE, LOCAL_SIZE, 1)]
void CubmapPrefilter(CSParam param)
{
    uint2 extent;
    uint layers;
    PrefilterMap.GetDimensions(extent.x, extent.y, layers);
    
    float3 colors[5] = {
        float3(0, 0, 0),
        float3(1, 0, 0),
        float3(0, 1, 0),
        float3(0, 0, 1),
        float3(1, 1, 1),
    };
    
    uint level = log2(PREFILTER_MAP_SIZE / extent.x);
    float roughness = float(level) / float(PREFILTER_MIP_LEVELS - 1);
    
    if (param.DispatchThreadID.x >= extent.x ||
        param.DispatchThreadID.y >= extent.y)
    {
        return;
    }
    
    uint2 size;
    Skybox.GetDimensions(size.x, size.y);
    uint resolution = size.x;
    
    float3 N = CalculateDirection(param.DispatchThreadID.z, param.DispatchThreadID.x, param.DispatchThreadID.y, extent);
    float3 R = N;
    float3 V = R;

    float3 prefilter_color = float3(0.0, 0.0, 0.0);
    float total_weight = 0.0;

    for (uint i = 0; i < SAMPLE_COUNT; i++)
    {
        float2 Xi = Hammersley(i, SAMPLE_COUNT);
        float3 H = GGXImportanceSampling(Xi, N, roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);

        float NoL = max(dot(N, L), 0.0);
        if (NoL > 0.0)
        {
            float D = GGX(N, H, roughness);
            float NoH = max(dot(N, H), 0.f);
            float HoV = max(dot(H, V), 0.f);
            float pdf = D * NoH / (4 * HoV) + 0.0001;
            
            float texel_ = 4.f * PI / (6.f * resolution * resolution);
            float sample_ = 1.f / (float(SAMPLE_COUNT) * pdf + 0.0001f);
            
            float mip_level = (roughness == 0.0 ? 0.0 : 0.5f * log2(sample_ / texel_));
            
            prefilter_color += Skybox.SampleLevel(SkyboxSampler, L, mip_level).rgb * NoL;
            total_weight += NoL;
        }
    }

    prefilter_color /= total_weight;
    PrefilterMap[int3(param.DispatchThreadID.xyz)] = float4(prefilter_color, 1.0);
    //PrefilterMap[int3(param.DispatchThreadID.xyz)] = float4(colors[level], 1.0);
}