#ifndef MATERIAL_HLSLI
#define MATERIAL_HLSLI

#include "Attribute.hlsli"
#include "Random.hlsli"
#include "Interaction.hlsli"
#include "MaterialResource.hlsli"

#include "Material/BSDF/BSDF.hlsli"

{{#MaterialHeaders}}
{{&MaterialHeader}}
{{/MaterialHeaders}}

struct MaterialData
{
    {{#Textures}}
    uint {{Texture}};
    {{/Textures}}
    {{#Samplers}}
    uint {{Sampler}};
    {{/Samplers}}
};

struct BSDF
{
    {{&BxDFType}} {{&BxDFName}};

    void Init(SurfaceInteraction surface_interaction)
    {
        MaterialData material_data = GetMaterialData < MaterialData > (surface_interaction.material);
        {{#Initializations}}
        {{&Initialization}}
        {{/Initializations}}
    }
        
     uint Flags()
     {
        return {{&BxDFName}}.Flags();
    }

    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        return {{&BxDFName}}.Eval(woW, wiW, mode);
    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        return {{&BxDFName}}.PDF(woW, wiW, mode, flags);
    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        return {{&BxDFName}}.Samplef(woW, uc, u, mode, flags);
    }

    float3 GetEmissive()
    {
        return {{&BxDFName}}.GetEmissive();
    }
};

struct Material
{
    BSDF bsdf;

    void Init(SurfaceInteraction surface_interaction)
    {
        bsdf.Init(surface_interaction);
    }
};

#endif