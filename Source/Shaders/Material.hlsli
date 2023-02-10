#ifndef MATERIAL_HLSLI
#define MATERIAL_HLSLI

#include "Attribute.hlsli"
#include "Random.hlsli"
#include "Interaction.hlsli"
#include "MaterialResource.hlsli"

#include "Material/BSDF/BSDF.hlsli"
#include "Material/Material/DiffuseMaterial.hlsli"
#include "Material/Material/ConductorMaterial.hlsli"
#include "Material/Material/DielectricMaterial.hlsli"
#include "Material/Material/ThinDielectricMaterial.hlsli"
#include "Material/Material/MixMaterial.hlsli"
#include "Material/Material/MaskedMaterial.hlsli"
#include "Material/Material/PrincipledMaterial.hlsli"

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