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

struct SH9
{
    float weights[9];
};

struct SH9Color
{
    float3 weights[9];
};

float AreaIntegration(float x, float y)
{
    return atan2(sqrt(x * x + y * y + 1), x * y);
}

SH9 ProjectSH9(float3 dir)
{
    SH9 sh;

    sh.weights[0] = 0.282095;

    sh.weights[1] = -0.488603 * dir.y;
    sh.weights[2] = 0.488603 * dir.z;
    sh.weights[3] = -0.488603 * dir.x;

    sh.weights[4] = 1.092548 * dir.x * dir.y;
    sh.weights[5] = -1.092548 * dir.y * dir.z;
    sh.weights[7] = 0.315392 * (3.0 * dir.z * dir.z - 1.0);
    sh.weights[6] = -1.092548 * dir.x * dir.z;
    sh.weights[8] = 0.546274 * (dir.x * dir.x - dir.y * dir.y);

    return sh;
}

float CalculateSolidAngle(uint x, uint y)
{
    float u = 2.0 * (float(x) + 0.5) / float(push_constants.extent.x) - 1.0;
    float v = 2.0 * (float(y) + 0.5) / float(push_constants.extent.y) - 1.0;

    float x0 = u - 1.0 / float(push_constants.extent.x);
    float x1 = u + 1.0 / float(push_constants.extent.x);
    float y0 = v - 1.0 / float(push_constants.extent.y);
    float y1 = v + 1.0 / float(push_constants.extent.y);

    return AreaIntegration(x0, y0) - AreaIntegration(x0, y1) - AreaIntegration(x1, y0) + AreaIntegration(x1, y1);
}

float3 CalculateDirection(uint face_idx, uint face_x, uint face_y)
{
    float u = 2.0 * (float(face_x) + 0.5) / float(push_constants.extent.x) - 1.0;
    float v = 2.0 * (float(face_y) + 0.5) / float(push_constants.extent.y) - 1.0;
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

    float3 dir = CalculateDirection(param.DispatchThreadID.z, param.DispatchThreadID.x, param.DispatchThreadID.y);
    float solid_angle = CalculateSolidAngle(param.DispatchThreadID.x, param.DispatchThreadID.y);
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