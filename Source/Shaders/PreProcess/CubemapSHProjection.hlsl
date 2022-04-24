#include "../SphericalHarmonic.hlsli"
#include "../Math.hlsli"

#define POSITIVE_X 0
#define NEGATIVE_X 1
#define POSITIVE_Y 2
#define NEGATIVE_Y 3
#define POSITIVE_Z 4
#define NEGATIVE_Z 5

#define LOCAL_SIZE 8

RWTexture2DArray<float4> SHIntermediate : register(u0);
TextureCube Skybox : register(t1);
SamplerState SkyboxSampler : register(s1);

[[vk::push_constant]]
struct
{
    uint2 extent;
} push_constants;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint3 GroupThreadID : SV_GroupThreadID;
};

groupshared SH9Color shared_sh_coeffs[LOCAL_SIZE][LOCAL_SIZE];
groupshared float shared_weights[LOCAL_SIZE][LOCAL_SIZE];

[numthreads(LOCAL_SIZE, LOCAL_SIZE, 1)]
void main(CSParam param)
{
    for (int i = 0; i < 9; i++)
    {
        shared_sh_coeffs[param.GroupThreadID.x][param.GroupThreadID.y].weights[i] = float3(0.0, 0.0, 0.0);
    }

    GroupMemoryBarrier();

    SH9 basis;

    float3 dir = CalculateCubemapDirection(param.DispatchThreadID.z, param.DispatchThreadID.x, param.DispatchThreadID.y, push_constants.extent.x, push_constants.extent.y);
    float solid_angle = CalculateSolidAngle(param.DispatchThreadID.x, param.DispatchThreadID.y, push_constants.extent.x, push_constants.extent.y);
    float3 texel = Skybox.SampleLevel(SkyboxSampler, dir, 0.0).rgb;

    basis = ProjectSH9(dir);

    shared_weights[param.GroupThreadID.x][param.GroupThreadID.y] = solid_angle;

    for (i = 0; i < 9; i++)
    {
        shared_sh_coeffs[param.GroupThreadID.x][param.GroupThreadID.y].weights[i] += texel * basis.weights[i] * solid_angle;
    }

    GroupMemoryBarrier();

	// Add up all coefficients and weights along the X axis
    if (param.GroupThreadID.x == 0)
    {
        for (int i = 1; i < LOCAL_SIZE; i++)
        {
            shared_weights[0][param.GroupThreadID.y] += shared_weights[i][param.GroupThreadID.y];
            for (int j = 0; j < 9; j++)
            {
                shared_sh_coeffs[0][param.GroupThreadID.y].weights[j] += shared_sh_coeffs[i][param.GroupThreadID.y].weights[j];
            }
        }
    }

    GroupMemoryBarrier();

	// Add up all coefficients and weights along the Y axis
    if (param.GroupThreadID.x == 0 && param.GroupThreadID.y == 0)
    {
        for (int i = 1; i < LOCAL_SIZE; i++)
        {
            shared_weights[0][0] += shared_weights[0][i];
            for (int j = 0; j < 9; j++)
            {
                shared_sh_coeffs[0][0].weights[j] += shared_sh_coeffs[0][i].weights[j];
            }
        }

        for (int j = 0; j < 9; j++)
        {
            int3 p = int3(push_constants.extent.x / LOCAL_SIZE * j + param.DispatchThreadID.x / LOCAL_SIZE, param.DispatchThreadID.y / LOCAL_SIZE, param.DispatchThreadID.z);
            
            SHIntermediate[p] = float4(shared_sh_coeffs[0][0].weights[j], shared_weights[0][0]);
        }
    }
}