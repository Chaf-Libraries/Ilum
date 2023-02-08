#ifndef MATERIAL_HLSLI
#define MATERIAL_HLSLI

#include "Attribute.hlsli"
#include "Random.hlsli"
#include "Interaction.hlsli"
#include "MaterialResource.hlsli"

#include "Material/BSDF/BSDF.hlsli"
// #include "Material/Material/Matte.hlsli"
// #include "Material/Material/Mirror.hlsli"
#include "Material/BSDF/MixBSDF.hlsli"
#include "Material/BSDF/DiffuseBSDF.hlsli"
// #include "Material/BSDF/PlasticBSDF.hlsli"
// #include "Material/BSDF/MetalBSDF.hlsli"
// #include "Material/BSDF/SubstrateBSDF.hlsli"
// #include "Material/Archive/GlassBSDF.hlsli"
// #include "Material/BSDF/DiffuseTransmissionBSDF.hlsli"
#include "Material/BSDF/DielectricBSDF.hlsli"
#include "Material/BSDF/ConductorBSDF.hlsli"
#include "Material/BSDF/ThinDielectricBSDF.hlsli"
// #include "Material/Archive/MetalBSDF.hlsli"
// #include "Material/BSDF/BlendBSDF.hlsli"

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