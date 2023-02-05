#ifndef DIFFUSE_BSDF_HLSLI
#define DIFFUSE_BSDF_HLSLI

#include "BSDF.hlsli"
#include "BlendBSDF.hlsli"
#include "LambertianReflectionBSDF.hlsli"
#include "OrenNayarBSDF.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct DiffuseBSDF
{
    LambertianReflectionBSDF lambertian;
    OrenNayarBSDF oren_nayar;
    float sigma;
    
    void Init(float3 R_, float sigma_, float3 normal_)
    {
        sigma = sigma_;
        if(sigma==0.0)
        {
            lambertian.Init(R_, normal_);
        }
        else
        {
            oren_nayar.Init(R_, sigma, normal_);
        }
    }
    
    uint Flags()
    {
        return BSDF_DiffuseReflection;
    }
    
    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        if (sigma == 0.0)
        {
            return lambertian.Eval(woW, wiW, mode);
        }
        else
        {
            return oren_nayar.Eval(woW, wiW, mode);
        }
        return 0.f;
    }
    
    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        if (sigma == 0.0)
        {
            return lambertian.PDF(woW, wiW, mode, flags);
        }
        else
        {
            return oren_nayar.PDF(woW, wiW, mode, flags);
        }
        return 0.f;
    }
    
    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        BSDFSample bsdf_sample;
        if (sigma == 0.0)
        {
            return lambertian.Samplef(woW, uc, u, mode, flags);
        }
        else
        {
            return oren_nayar.Samplef(woW, uc, u, mode, flags);
        }
        bsdf_sample.Init();
        return bsdf_sample;
    }
};

#endif