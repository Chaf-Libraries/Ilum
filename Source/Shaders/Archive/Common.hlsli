#ifndef __COMMON_HLSL__
#define __COMMON_HLSL__

#include "ShaderInterop.hpp"

static const uint MetalRoughnessWorkflow = 0;
static const uint SpecularGlossinessWorkflow = 1;

#define MAX_TEXTURE_ARRAY_SIZE 1024

void Swap(inout float lhs, inout float rhs)
{
    float tmp = lhs;
    lhs = rhs;
    rhs = tmp;
}

float2 OctWrap(float2 v)
{
    return float2((1.0f - abs(v.y)) * (v.x >= 0.0f ? 1.0f : -1.0f), (1.0f - abs(v.x)) * (v.y >= 0.0f ? 1.0f : -1.0f));
}

float2 PackNormal(float3 n)
{
    float2 p = float2(n.x, n.y) * (1.0f / (abs(n.x) + abs(n.y) + abs(n.z)));
    p = (n.z < 0.0f) ? OctWrap(p) : p;
    return p;
}

float3 UnPackNormal(float2 p)
{
    float3 n = float3(p.x, p.y, 1.0f - abs(p.x) - abs(p.y));
    float2 tmp = (n.z < 0.0f) ? OctWrap(float2(n.x, n.y)) : float2(n.x, n.y);
    n.x = tmp.x;
    n.y = tmp.y;
    return normalize(n);
}

#define SRGB_FAST_APPROXIMATION 1
// sRGB to linear approximation
// see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
//-----------------------------------------------------------------------
float4 SRGBtoLINEAR(float4 srgbIn)
{
#ifdef SRGB_FAST_APPROXIMATION
    float3 linOut = pow(srgbIn.xyz, float3(2.2, 2.2, 2.2));
#else  //SRGB_FAST_APPROXIMATION
    float3 bLess = step(float3(0.04045, 0.04045, 0.04045), srgbIn.xyz);
    float3 linOut = lerp(srgbIn.xyz / float3(12.92, 12.92, 12.92), pow((srgbIn.xyz + float3(0.055, 0.055, 0.055)) / float3(1.055, 1.055, 1.055), float3(2.4, 2.4, 2.4)), bLess);
#endif  //SRGB_FAST_APPROXIMATION
    return float4(linOut, srgbIn.w);
}

#endif