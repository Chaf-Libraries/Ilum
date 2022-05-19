#ifndef __COMMON_HLSL__
#define __COMMON_HLSL__

#include "ShaderInterop.hpp"

ConstantBuffer<Instance> instances[] : register(b0, space1);
StructuredBuffer<Meshlet> meshlets[] : register(t1, space1);
StructuredBuffer<Vertex> vertices[] : register(t2, space1);
StructuredBuffer<uint> meshlet_vertices[] : register(t3, space1);
StructuredBuffer<uint> meshlet_primitives[] : register(t4, space1);
ConstantBuffer<Material> materials[] : register(b0, space2);
Texture2D<float4> texture_array[] : register(t1, space2);
SamplerState texture_sampler : register(s2, space2);

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