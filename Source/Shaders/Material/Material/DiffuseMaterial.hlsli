#ifndef DIFFUSE_MATERIAL_HLSLI
#define DIFFUSE_MATERIAL_HLSLI

#include "../BSDF/DiffuseBSDF.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct DiffuseMaterial
{
    DiffuseBSDF diffuse;
    Frame frame;

    void Init(float3 R, float3 normal)
    {
        diffuse.Init(SRGBtoLINEAR(R));
        frame.FromZ(normal);
    }
    
    uint Flags()
    {
        return diffuse.Flags();
    }

    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        return diffuse.Eval(frame.ToLocal(woW), frame.ToLocal(wiW), mode);
    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        return diffuse.PDF(frame.ToLocal(woW), frame.ToLocal(wiW), mode, flags);
    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        BSDFSample bsdf_sample = diffuse.Samplef(frame.ToLocal(woW), uc, u, mode, flags);
        bsdf_sample.wiW = frame.ToWorld(bsdf_sample.wi);
        return bsdf_sample;
    }
};

#endif