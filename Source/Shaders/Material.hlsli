#ifndef __MATERIAL_HLSL__
#define __MATERIAL_HLSL__

#include "BxDF.hlsli"
#include "Common.hlsli"

////////////// Matte Material //////////////
struct MatteMaterial
{
    float3 Kd;
    float sigma;
    
    LambertianReflection lambertian;
    OrenNayar oren_nayar;
    
    void Init(Material mat)
    {
        Kd = mat.base_color.rgb;
        sigma = mat.roughness;
        
        lambertian.R = Kd;
        oren_nayar.Init(Kd, sigma);
    }
    
    float3 f(float3 wo, float3 wi)
    {
        if (wo.z == 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }

        if (sigma == 0)
        {
            return lambertian.f(wo, wi);
        }
        else
        {
            return oren_nayar.f(wo, wi);
        }
    }
    
    float3 Samplef(float3 wo, Sampler _sampler, out float3 wi, out float pdf)
    {
        if (wo.z == 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }

        if (sigma == 0)
        {
            return lambertian.Samplef(wo, _sampler, wi, pdf);
        }
        else
        {
            return oren_nayar.Samplef(wo, _sampler, wi, pdf);
        }
    }
};

////////////// BSDF //////////////
struct BSDF
{
    Material mat;
    
    void Init(Material mat_)
    {
        mat = mat_;
    }
    
    float3 f(float3 wo, float3 wi)
    {
        if (mat.material_type == BxDF_Matte)
        {
            MatteMaterial bsdf;
            bsdf.Init(mat);
            return bsdf.f(wo, wi);
        }
        
        return float3(0.0, 0.0, 0.0);
    }
    
    float3 Samplef(float3 wo, Sampler _sampler, out float3 wi, out float pdf)
    {
        if (mat.material_type == BxDF_Matte)
        {
            MatteMaterial bsdf;
            bsdf.Init(mat);
            return bsdf.Samplef(wo, _sampler, wi, pdf);
        }
        
        return float3(0.0, 0.0, 0.0);
    }
};

#endif