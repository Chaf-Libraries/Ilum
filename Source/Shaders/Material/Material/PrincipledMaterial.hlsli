#ifndef PRINCIPLED_MATERIAL_HLSLI
#define PRINCIPLED_MATERIAL_HLSLI

#include "../BSDF/PrincipledBSDF.hlsli"
#include "../Scattering.hlsli"
#include "../Fresnel.hlsli"
#include "../../Math.hlsli"
#include "../../Common.hlsli"
#include "../../Interaction.hlsli"

struct PrincipledMaterial
{
    PrincipledBSDF principled;
    Frame frame;

    void Init(
        float3 base_color,
        float metallic,
        float roughness,
        float anisotropic,
        float sheen,
        float sheen_tint,
        float specular,
        float spec_tint,
        float clearcoat,
        float clearcoat_gloss,
        float flatness,
        float spec_trans,
        float eta,
        float3 emissive,
        bool twoside,
        float3 normal)
    {
        frame.FromZ(normal);
        principled.Init(base_color, roughness, anisotropic, 
            spec_trans, eta, sheen, sheen_tint, specular, spec_tint, metallic, clearcoat,
            clearcoat_gloss, flatness, emissive, twoside);
    }

    float3 GetEmissive()
    {
        return principled.GetEmissive();
    }


    uint Flags()
    {
        return principled.Flags();
    }

    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        return principled.Eval(frame.ToLocal(woW), frame.ToLocal(wiW), mode);
    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        return principled.PDF(frame.ToLocal(woW), frame.ToLocal(wiW), mode, flags);
    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        BSDFSample bsdf_sample = principled.Samplef(frame.ToLocal(woW), uc, u, mode, flags);
        bsdf_sample.wiW = frame.ToWorld(bsdf_sample.wi);
        return bsdf_sample;
    }
};

#endif