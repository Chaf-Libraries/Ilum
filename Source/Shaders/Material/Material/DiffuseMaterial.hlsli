#ifndef DIFFUSE_MATERIAL_HLSLI
#define DIFFUSE_MATERIAL_HLSLI

#include "../BSDF/DiffuseBSDF.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct DiffuseMaterial
{
    DiffuseBSDF diffuse;
    Frame frame;

    GBufferData GetGBufferData()
    {
        GBufferData data;
        data.albedo = diffuse.R;
        data.metallic = 0.f;
        data.roughness = 1.f;
        data.anisotropic = 0.f;
        data.normal = frame.z;
        data.emissive = 0.f;
        return data;
    }
    
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
      //  return frame.z;
        return diffuse.Eval(frame.ToLocal(woW), frame.ToLocal(wiW), mode) * abs(dot(wiW, frame.z));
    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        return diffuse.PDF(frame.ToLocal(woW), frame.ToLocal(wiW), mode, flags);
    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        BSDFSample bsdf_sample = diffuse.Samplef(frame.ToLocal(woW), uc, u, mode, flags);
        bsdf_sample.wiW = frame.ToWorld(bsdf_sample.wi);
        bsdf_sample.f *= abs(dot(bsdf_sample.wiW, frame.z));
        return bsdf_sample;
    }
};

#endif