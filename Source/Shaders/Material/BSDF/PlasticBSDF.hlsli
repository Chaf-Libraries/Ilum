#ifndef PLASTIC_BSDF_HLSLI
#define PLASTIC_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../Scattering.hlsli"
#include "../Fresnel.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct PlasticBSDF
{
    float3 diff;
    float3 spec;
    float eta;
    Frame frame;

    void Init(float3 diff_, float3 spec_, float eta_, float3 normal)
    {
        diff = diff_;
        spec = spec_;
        eta = eta_;
        frame.FromZ(normal);
    }

    uint Flags()
    {
        return BSDF_Glossy | BSDF_Reflection | BSDF_Diffuse;
    }
};

#endif