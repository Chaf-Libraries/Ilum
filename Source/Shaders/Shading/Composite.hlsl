#include "../Common.hlsli"

RWTexture2D<float4> Output;

#ifndef RUNTIME
#define HAS_LIGHT_DIRECT_ILLUMINATION
#define HAS_ENV_DIRECT_ILLUMINATION
#define HAS_AMBIENT_OCCLUSION
#define HAS_ENVIRONMENT
#endif

#ifdef HAS_AMBIENT_OCCLUSION
Texture2D<float> AmbientOcclusion;
#endif

#ifdef HAS_LIGHT_DIRECT_ILLUMINATION
Texture2D<float4> LightDirectIllumination;
#endif

#ifdef HAS_ENV_DIRECT_ILLUMINATION
Texture2D<float4> EnvDirectIllumination;
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
    float3 environment = 0.f;
#ifdef HAS_ENVIRONMENT
    environment = Environment[param.DispatchThreadID.xy].xyz;
#endif
    
    if (IsBlack(environment))
    {
#ifdef HAS_LIGHT_DIRECT_ILLUMINATION
        result += LightDirectIllumination[param.DispatchThreadID.xy].xyz;
#endif
        
#ifdef HAS_ENV_DIRECT_ILLUMINATION
        float3 env_direct_illumination = EnvDirectIllumination[param.DispatchThreadID.xy].xyz;
#ifdef HAS_AMBIENT_OCCLUSION
        env_direct_illumination *= AmbientOcclusion[param.DispatchThreadID.xy].r;
#endif
        result += env_direct_illumination;
#endif
    }
    else
    {
        result += environment;
    }
    
    Output[param.DispatchThreadID.xy] = float4(result, 1.f);
}