#ifndef DIELECTRIC_MATERIAL_HLSLI
#define DIELECTRIC_MATERIAL_HLSLI

#include "../BSDF/DielectricBSDF.hlsli"

struct DielectricMaterial
{
    DielectricBSDF dielectric;
    Frame frame;

    HAS_NO_EMISSIVE

    void Init(float3 R, float3 T, float ior, float roughness, float3 normal)
    {
        dielectric.Init(SRGBtoLINEAR(R), SRGBtoLINEAR(T), ior, roughness);
        frame.FromZ(normal);
    }

    uint Flags()
    {
        return dielectric.Flags();
    }

    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        return dielectric.Eval(frame.ToLocal(woW), frame.ToLocal(wiW), mode) * abs(dot(wiW, frame.z));
    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        return dielectric.PDF(frame.ToLocal(woW), frame.ToLocal(wiW), mode, flags);
    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        BSDFSample bsdf_sample = dielectric.Samplef(frame.ToLocal(woW), uc, u, mode, flags);
        bsdf_sample.wiW = frame.ToWorld(bsdf_sample.wi);
        bsdf_sample.f *= abs(dot(bsdf_sample.wiW, frame.z));
        return bsdf_sample;
    }
};

#endif