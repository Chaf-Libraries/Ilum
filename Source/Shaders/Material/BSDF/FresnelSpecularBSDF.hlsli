#ifndef FRESNEL_SPECULAR_BSDF_HLSLI
#define FRESNEL_SPECULAR_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../Scattering.hlsli"
#include "../Fresnel.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct FresnelSpecularBSDF
{
    float3 R, T;
    float etaA, etaB;
    Frame frame;
    
    void Init(float3 R_, float3 T_, float etaA_, float etaB_, float3 normal_)
    {
        R = R_;
        T = T_;
        etaA = etaA_;
        etaB = etaB_;
        frame.FromZ(normal_);
    }

    uint Flags()
    {
        return BSDF_Reflection | BSDF_Specular | BSDF_Transmission;
    }
    
    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        return 0.f;
    }
    
    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        return 0.f;
    }
    
    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        
        BSDFSample bsdf_sample;
        
        bsdf_sample.Init();
        
        if (!(flags & SampleFlags_Transmission)||
            !(flags & SampleFlags_Reflection))
        {
            return bsdf_sample;
        }
        
        FresnelDielectric fresnel;
        fresnel.eta_i = etaA;
        fresnel.eta_t = etaB;
        float3 F = fresnel.Eval(CosTheta(wo));
        
        if (u.x < F.x)
        {
            float3 wi = float3(-wo.x, -wo.y, wo.z);
            
            float3 wiW = frame.ToWorld(wi);
        
            bsdf_sample.f = F * R / AbsCosTheta(wi);
            bsdf_sample.wiW = wiW;
            bsdf_sample.pdf = F.x;
            bsdf_sample.flags = BSDF_SpecularReflection;
            bsdf_sample.eta = 1;
            
            return bsdf_sample;
        }
        else
        {
            bool entering = CosTheta(wo) > 0.0;
            float etaI = entering ? etaA : etaB;
            float etaT = entering ? etaB : etaA;
            float3 wi = 0.f;
            
            if (!Refract(wo, Faceforward(float3(0.0, 0.0, 1.0), wo), etaI / etaT, wi))
            {
                return bsdf_sample;
            }

            float3 ft = T * (float3(1.0, 1.0, 1.0) - F);

            if (mode == TransportMode_Radiance)
            {
                ft *= (etaI * etaI) / (etaT * etaT);
            }
            
            float3 wiW = frame.ToWorld(wi);
        
            bsdf_sample.f = ft / AbsCosTheta(wi);
            bsdf_sample.wiW = wiW;
            bsdf_sample.pdf = 1.0 - F.x;
            bsdf_sample.flags = BSDF_SpecularTransmission;
            bsdf_sample.eta = etaI / etaT;
            
            return bsdf_sample;
        }
        
        return bsdf_sample;
    }
};

#endif