#include "../Constants.hlsli"

#define SH_INTERMEDIATE_SIZE 1024 / 8
#define CUBEMAP_FACE_NUM 6

RWTexture2D<float4> IrradianceSH : register(u0);
Texture2DArray SHIntermediate : register(t1);
SamplerState SHSampler : register(s1);

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

groupshared float3 shared_sh_coeffs[SH_INTERMEDIATE_SIZE][CUBEMAP_FACE_NUM];
groupshared float shared_weights[SH_INTERMEDIATE_SIZE][CUBEMAP_FACE_NUM];

[numthreads(1, SH_INTERMEDIATE_SIZE, CUBEMAP_FACE_NUM)]
void main(CSParam param)
{
    shared_sh_coeffs[param.DispatchThreadID.y][param.DispatchThreadID.z] = float3(0.0, 0.0, 0.0);
    shared_weights[param.DispatchThreadID.y][param.DispatchThreadID.z] = 0.0;

    GroupMemoryBarrier();

    for (uint i = 0; i < SH_INTERMEDIATE_SIZE; i++)
    {
        int3 p = int3(param.DispatchThreadID.x * SH_INTERMEDIATE_SIZE + i, param.DispatchThreadID.y, param.DispatchThreadID.z);
        float4 val = SHIntermediate.Load(int4(p, 0.0));

        shared_sh_coeffs[param.DispatchThreadID.y][param.DispatchThreadID.z] += val.rgb;
        shared_weights[param.DispatchThreadID.y][param.DispatchThreadID.z] += val.a;
    }

    GroupMemoryBarrier();

    if (param.DispatchThreadID.z == 0)
    {
        for (uint i = 1; i < CUBEMAP_FACE_NUM; i++)
        {
            shared_sh_coeffs[param.DispatchThreadID.y][0] += shared_sh_coeffs[param.DispatchThreadID.y][i];
            shared_weights[param.DispatchThreadID.y][0] += shared_weights[param.DispatchThreadID.y][i];
        }
    }

    GroupMemoryBarrier();

    if (param.DispatchThreadID.y == 0 && param.DispatchThreadID.z == 0)
    {
        for (uint i = 0; i < SH_INTERMEDIATE_SIZE; i++)
        {
            shared_sh_coeffs[0][0] += shared_sh_coeffs[i][0];
            shared_weights[0][0] += shared_weights[i][0];
        }

        float scale = (4.0 * PI) / shared_weights[0][0];

        IrradianceSH[int2(param.DispatchThreadID.x, 0)] = float4(shared_sh_coeffs[0][0] * scale, shared_weights[0][0]);     
    }
}