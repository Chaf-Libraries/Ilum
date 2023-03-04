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