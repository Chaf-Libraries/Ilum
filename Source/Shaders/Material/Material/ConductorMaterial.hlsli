#ifndef CONDUCTOR_MATERIAL_HLSLI
#define CONDUCTOR_MATERIAL_HLSLI

#include "../BSDF/ConductorBSDF.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct ConductorMaterial
{
    ConductorBSDF conductor;
    Frame frame;

    HAS_NO_EMISSIVE

    void Init(float3 R, float roughness, float3 eta, float3 k, float3 normal)
    {
        conductor.Init(SRGBtoLINEAR(R), roughness, eta, k);
        frame.FromZ(normal);
    }
    
    uint Flags()
    {
        return conductor.Flags();
    }

    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        return conductor.Eval(frame.ToLocal(woW), frame.ToLocal(wiW), mode) * abs(dot(wiW, frame.z));
    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        return conductor.PDF(frame.ToLocal(woW), frame.ToLocal(wiW), mode, flags);
    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        BSDFSample bsdf_sample = conductor.Samplef(frame.ToLocal(woW), uc, u, mode, flags);
        bsdf_sample.wiW = frame.ToWorld(bsdf_sample.wi);
        bsdf_sample.f *= abs(dot(bsdf_sample.wiW, frame.z));
        return bsdf_sample;
    }
};

#endif