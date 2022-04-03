#ifndef __MATH_HLSL__
#define __MATH_HLSL__

void CreateCoordinateSystem(in float3 N, out float3 Nt, out float3 Nb)
{
    Nt = normalize(((abs(N.z) > 0.99999f) ? float3(-N.x * N.y, 1.0f - N.y * N.y, -N.y * N.z) :
                                            float3(-N.x * N.z, -N.y * N.z, 1.0f - N.z * N.z)));
    Nb = cross(Nt, N);
}

#endif