#include "../Common.hlsli"

RWTexture2D<float4> Output;

#ifndef RUNTIME
#define HAS_DIRECT_ILLUMINATION
#define HAS_ENVIRONMENT
#endif

#ifdef HAS_DIRECT_ILLUMINATION
Texture2D<float4> DirectIllumination;
#endif

#ifdef HAS_ENVIRONMENT
Texture2D<float4> Environment;
#endif

[numthreads(8, 8, 1)]
void CSmain(CSParam param)
{
    uint2 extent;
    Output.GetDimensions(extent.x, extent.y);
    
    if (param.DispatchThreadID.x >= extent.x ||
        param.DispatchThreadID.y >= extent.y)
    {
        return;
    }
    
    float3 result = 0.f;
    
#ifdef HAS_ENVIRONMENT
    float3 environment = Environment[param.DispatchThreadID.xy].xyz;
#endif
    
    if (IsBlack(environment))
    {
#ifdef HAS_DIRECT_ILLUMINATION
        result += DirectIllumination[param.DispatchThreadID.xy].xyz;
#endif
    }
    else
    {
        result += environment;
    }
    
    Output[param.DispatchThreadID.xy] = float4(result, 1.f);
}