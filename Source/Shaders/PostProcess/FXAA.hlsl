#include "../Common.hlsli"

struct Config
{
    float fixed_threshold;
    float relative_threshold;
    float subpixel_blending;
};

Texture2D Input;
SamplerState Sampler;
RWTexture2D<float4> Output;
ConstantBuffer<Config> ConfigBuffer;

#ifdef FXAA_QUALITY_LOW
#define EXTRA_EDGE_STEPS 3
#define EDGE_STEP_SIZES 1.5, 2.0, 2.0
#define LAST_EDGE_STEP_GUESS 8.0
#elif FXAA_QUALITY_MEDIUM
#define EXTRA_EDGE_STEPS 8
#define EDGE_STEP_SIZES 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0
#define LAST_EDGE_STEP_GUESS 8.0
#else   // FXAA_QUALITY_HIGH
#define EXTRA_EDGE_STEPS 10
#define EDGE_STEP_SIZES 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0
#define LAST_EDGE_STEP_GUESS 8.0
#endif

static const float edgeStepSizes[EXTRA_EDGE_STEPS] = { EDGE_STEP_SIZES };

float GetLuma(float2 uv, float2 offset, float2 texel_size)
{
    uv += float2(offset) * texel_size;
    return sqrt(Luminance(Input.SampleLevel(Sampler, uv, 0.0).rgb));
}

// +1  NW   N     NE
// +0  W     M      E
// -1   SW    S      SE
//      -1      +0    +1
struct LumaNeighborhood
{
    float m, n, e, s, w, ne, se, sw, nw;
    float highest, lowest, range;
};

struct FXAAEdge
{
    bool is_horizontal;
    float pixel_step;
    float luma_gradient, luma;
};

LumaNeighborhood GetLumaNeighborhood(float2 uv, float2 texel_size)
{
    LumaNeighborhood luma;
    luma.m = GetLuma(uv, float2(0.0, 0.0), texel_size);
    luma.n = GetLuma(uv, float2(0.0, 1.0), texel_size);
    luma.e = GetLuma(uv, float2(1.0, 0.0), texel_size);
    luma.s = GetLuma(uv, float2(0.0, -1.0), texel_size);
    luma.w = GetLuma(uv, float2(-1.0, 0.0), texel_size);
    luma.ne = GetLuma(uv, float2(1.0, 1.0), texel_size);
    luma.se = GetLuma(uv, float2(1.0, -1.0), texel_size);
    luma.sw = GetLuma(uv, float2(-1.0, -1.0), texel_size);
    luma.nw = GetLuma(uv, float2(-1.0, 1.0), texel_size);
    luma.highest = max(luma.m, max(luma.n, max(luma.e, max(luma.s, luma.w))));
    luma.lowest = min(luma.m, min(luma.n, min(luma.e, min(luma.s, luma.w))));
    luma.range = luma.highest - luma.lowest;
    return luma;
}

float GetSubpixelBlendFactor(LumaNeighborhood luma)
{
    float filter = luma.n + luma.e + luma.s + luma.w;
    filter = 2.0 * filter + luma.ne + luma.se + luma.sw + luma.nw;
    filter *= 1.0 / 12.0;
    filter = saturate(filter / luma.range);
    filter = smoothstep(0.0, 1.0, filter);
    return filter * filter * ConfigBuffer.subpixel_blending;
}

bool FXAASkip(LumaNeighborhood luma)
{
    return luma.range < max(ConfigBuffer.fixed_threshold, ConfigBuffer.relative_threshold * luma.highest);
}

bool IsHorizontalEdge(LumaNeighborhood luma)
{
    float horizontal = 2.0 * abs(luma.n + luma.s - 2.0 * luma.m) + abs(luma.nw + luma.sw - 2.0 * luma.w) + abs(luma.ne + luma.se - 2.0 * luma.e);
    float vertical = 2.0 * abs(luma.e + luma.w - 2.0 * luma.m) + abs(luma.ne + luma.nw - 2.0 * luma.n) + abs(luma.se + luma.sw - 2.0 * luma.s);
    return horizontal >= vertical;
}

FXAAEdge GetFXAAEdge(LumaNeighborhood luma, float2 texel_size)
{
    FXAAEdge edge;
    edge.is_horizontal = IsHorizontalEdge(luma);
    float lumaP, lumaN;
    if (edge.is_horizontal)
    {
        edge.pixel_step = texel_size.y;
        lumaP = luma.n;
        lumaN = luma.s;
    }
    else
    {
        edge.pixel_step = texel_size.x;
        lumaP = luma.e;
        lumaN = luma.w;
    }
    
    float gradientP = abs(lumaP - luma.m);
    float gradientN = abs(lumaN - luma.m);
    
    if (gradientP < gradientN)
    {
        edge.pixel_step = -edge.pixel_step;
        edge.luma_gradient = gradientN;
        edge.luma = lumaN;
    }
    else
    {
        edge.luma_gradient = gradientP;
        edge.luma = lumaP;
    }
    
    return edge;
}

float GetEdgeBlendFactor(LumaNeighborhood luma, FXAAEdge edge, float2 uv, float2 texel_size)
{
    float2 edge_uv = uv;
    float2 uv_step = float2(0.0, 0.0);
    
    if (edge.is_horizontal)
    {
        edge_uv.y += 0.5 * edge.pixel_step;
        uv_step.x = texel_size.x;
    }
    else
    {
        edge_uv.x += 0.5 * edge.pixel_step;
        uv_step.y = texel_size.y;
    }
    
    float edge_luma = 0.5 * (luma.m + edge.luma);
    float gradient_threshold = 0.25 * edge.luma_gradient;
    
    float2 uvP = edge_uv + uv_step;
    float luma_deltaP = GetLuma(uvP, float2(0, 0), texel_size) - edge_luma;
    bool atP = abs(luma_deltaP) >= gradient_threshold;
    
    for (int i = 0; i < EXTRA_EDGE_STEPS && !atP; i++)
    {
        uvP += uv_step * edgeStepSizes[i];
        luma_deltaP = GetLuma(uvP, float2(0, 0), texel_size) - edge_luma;
        atP = abs(luma_deltaP) >= gradient_threshold;
    }
    if (!atP)
    {
        uvP += uv_step * LAST_EDGE_STEP_GUESS;
    }
    
    float2 uvN = edge_uv - uv_step;
    float luma_deltaN = GetLuma(uvN, float2(0, 0), texel_size) - edge_luma;
    bool atN = abs(luma_deltaN) >= gradient_threshold;
    
    for (int i = 0; i < EXTRA_EDGE_STEPS && !atN; i++)
    {
        uvN -= uv_step * edgeStepSizes[i];
        luma_deltaN = GetLuma(uvN, float2(0, 0), texel_size) - edge_luma;
        atN = abs(luma_deltaN) >= gradient_threshold;
    }
    if (!atN)
    {
        uvN -= uv_step * LAST_EDGE_STEP_GUESS;
    }
    
    float dist_to_P, dist_to_N;
    if (edge.is_horizontal)
    {
        dist_to_P = uvP.x - uv.x;
        dist_to_N = uv.x - uvN.x;
    }
    else
    {
        dist_to_P = uvP.y - uv.y;
        dist_to_N = uv.y - uvN.y;
    }
    
    float dist;
    bool delta_sign;
    
    if (dist_to_P <= dist_to_N)
    {
        dist = dist_to_P;
        delta_sign = luma_deltaP >= 0;
    }
    else
    {
        dist = dist_to_N;
        delta_sign = luma_deltaN >= 0;
    }
    
    if (delta_sign == (luma.m - edge_luma) > 0)
    {
        return 0.0;
    }
    
    return 0.5 - dist / (dist_to_N + dist_to_P);
}

[numthreads(8, 8, 1)]
void CSmain(CSParam param)
{
    uint2 extent;
    Input.GetDimensions(extent.x, extent.y);
    float2 texel_size = 1.0 / float2(extent);
    
    float2 uv = (float2(param.DispatchThreadID.xy) + float2(0.5, 0.5)) * texel_size;
    
    if (param.DispatchThreadID.x >= extent.x || param.DispatchThreadID.y >= extent.y)
    {
        return;
    }
    
    LumaNeighborhood luma = GetLumaNeighborhood(uv, texel_size);
    
    if (FXAASkip(luma))
    {
        Output[param.DispatchThreadID.xy] = float4(Input[param.DispatchThreadID.xy].rgb, 1.0);
        return;
    }
    
    FXAAEdge edge = GetFXAAEdge(luma, texel_size);
    float blend_factor = max(GetSubpixelBlendFactor(luma), GetEdgeBlendFactor(luma, edge, uv, texel_size));
    
    if (edge.is_horizontal)
    {
        uv.y += blend_factor * edge.pixel_step;
    }
    else
    {
        uv.x += blend_factor * edge.pixel_step;
    }
    
    Output[param.DispatchThreadID.xy] = float4(Input.SampleLevel(Sampler, uv, 0.0).rgb, 1.0);
}