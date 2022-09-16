#include "Tonemapper.hlsli"

Texture2D InImage : register(t0);
SamplerState TexSampler : register(s1);
RWTexture2D<float4> OutImage : register(u2);

#ifndef FXAA_REDUCE_MIN
    #define FXAA_REDUCE_MIN   (1.0/ 128.0)
#endif
#ifndef FXAA_REDUCE_MUL
    #define FXAA_REDUCE_MUL   (1.0 / 8.0)
#endif
#ifndef FXAA_SPAN_MAX
    #define FXAA_SPAN_MAX     8.0
#endif

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

[numthreads(8, 8, 1)]
void main(CSParam param)
{
    uint2 extent;
    InImage.GetDimensions(extent.x, extent.y);
    float2 texel_size = 1.0 / float2(extent);
        
    if (param.DispatchThreadID.x >= extent.x || param.DispatchThreadID.y >= extent.y)
    {
        return;
    }
    
    float4 color;
    float3 rgbNW = InImage.SampleLevel(TexSampler, (float2(param.DispatchThreadID.xy) + float2(-1, -1)) * texel_size, 0.0).rgb;
    float3 rgbNE = InImage.SampleLevel(TexSampler, (float2(param.DispatchThreadID.xy) + float2(1, -1)) * texel_size, 0.0).rgb;
    float3 rgbSW = InImage.SampleLevel(TexSampler, (float2(param.DispatchThreadID.xy) + float2(-1, 1)) * texel_size, 0.0).rgb;
    float3 rgbSE = InImage.SampleLevel(TexSampler, (float2(param.DispatchThreadID.xy) + float2(1, 1)) * texel_size, 0.0).rgb;
    float4 tex_color = InImage.SampleLevel(TexSampler, float2(param.DispatchThreadID.xy) * texel_size, 0.0).rgba;
    float3 rgbM = tex_color.rgb;
    float3 luma = float3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM = dot(rgbM, luma);
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    
    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));
    
    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) *
                          (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
    
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(float2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
              max(float2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
              dir * rcpDirMin)) * texel_size;
    
    float3 rgbA = 0.5 * (
        InImage.SampleLevel(TexSampler, float2(param.DispatchThreadID.xy) * texel_size + dir * (1.0 / 3.0 - 0.5), 0.0).xyz +
        InImage.SampleLevel(TexSampler, float2(param.DispatchThreadID.xy) * texel_size + dir * (2.0 / 3.0 - 0.5), 0.0).xyz);
    
    float3 rgbB = rgbA * 0.5 + 0.25 * (
        InImage.SampleLevel(TexSampler, float2(param.DispatchThreadID.xy) * texel_size + dir * -0.5, 0.0).xyz +
        InImage.SampleLevel(TexSampler, float2(param.DispatchThreadID.xy) * texel_size + dir * 0.5, 0.0).xyz);

    float lumaB = dot(rgbB, luma);
    if ((lumaB < lumaMin) || (lumaB > lumaMax))
    {
        color = float4(rgbA, tex_color.a);
    }
    else
    {
        color = float4(rgbB, tex_color.a);
    }
    
    OutImage[param.DispatchThreadID.xy] = color;

}