#ifndef THIN_DIELECTRIC_MATERIAL_HLSLI
#define THIN_DIELECTRIC_MATERIAL_HLSLI

#include "../BSDF/ThinDielectricBSDF.hlsli"

struct ThinDielectricMaterial
{
    ThinDielectricBSDF thin_dielectric;
    Frame frame;

    void Init(float3 R, float3 T, float eta, float3 normal)
    {
        thin_dielectric.Init(SRGBtoLINEAR(R), SRGBtoLINEAR(T), eta);
        frame.FromZ(normal);
    }

    uint Flags()
    {
        return thin_dielectric.Flags();
    }

    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        return thin_dielectric.Eval(frame.ToLocal(woW), frame.ToLocal(wiW), mode);
    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        return thin_dielectric.PDF(frame.ToLocal(woW), frame.ToLocal(wiW), mode, flags);
    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        BSDFSample bsdf_sample = thin_dielectric.Samplef(frame.ToLocal(woW), uc, u, mode, flags);
        bsdf_sample.wiW = frame.ToWorld(bsdf_sample.wi);
        return bsdf_sample;
    }
};

#endif