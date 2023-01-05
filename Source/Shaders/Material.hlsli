#ifndef MATERIAL_HLSLI
#define MATERIAL_HLSLI

#include "Attribute.hlsli"
#include "Random.hlsli"

#include "Material/BSDF/BSDF.hlsli"

{{#Textures}}
Texture2D<float4> {{Texture}};
{{/Textures}}

{{#Samplers}}
SamplerState {{Sampler}};
{{/Samplers}}

struct BSDF
{
    {{&BxDFType}} {{&BxDFName}}

    void Init()
    {
        {{#Initializations}}
        {{&Initialization}}
        {{/Initializations}}
    }

    float3 Eval(float3 wo, float3 wi, TransportMode mode)
    {
        return {{&BxDFName}}.Eval(wo, wi, mode);
    }

    float PDF(float3 wo, float3 wi, TransportMode mode, SampleFlags flags)
    {
        return {{&BxDFName}}.PDF(wo, wi, mode, flags);
    }

    BSDFSample Samplef(float3 wo, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        return {{&BxDFName}}.Samplef(wo, uc, u, mode, flags);
    }
}

struct Material
{
    BSDF bsdf;

    void Init()
    {
        bsdf.Init();
    }
}

#endif