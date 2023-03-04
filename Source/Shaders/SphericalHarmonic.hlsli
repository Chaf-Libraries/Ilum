#ifndef __SPHERICALHARMONIC_HLSL__
#define __SPHERICALHARMONIC_HLSL__

#include "Math.hlsli"

static const float CosineA0 = PI;
static const float CosineA1 = (2.0 * PI) / 3.0;
static const float CosineA2 = PI * 0.25;

struct SH9
{
    float weights[9];
};

struct SH9Color
{
    float3 weights[9];
};

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

SH9 EvaluateSH(float3 dir)
{
    SH9 basis = ProjectSH9(dir);

    basis.weights[0] *= CosineA0;

    basis.weights[1] *= CosineA1;
    basis.weights[2] *= CosineA1;
    basis.weights[3] *= CosineA1;

    basis.weights[4] *= CosineA2;
    basis.weights[5] *= CosineA2;
    basis.weights[6] *= CosineA2;
    basis.weights[7] *= CosineA2;
    basis.weights[8] *= CosineA2;

    return basis;
}

#endif