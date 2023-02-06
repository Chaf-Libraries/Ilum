#ifndef FRESNEL_BLEND_HLSLI
#define FRESNEL_BLEND_HLSLI

#include "BSDF.hlsli"
#include "MicrofacetReflectionBSDF.hlsli"
#include "../Fresnel.hlsli"
#include "../Scattering.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

template<typename Distribution>
struct FresnelBlendBSDF
{
    float3 Rd, Rs;
    Distribution distribution;
    Frame frame;
    
    float3 SchlickFresnel(float cos_theta)
    {
        return Rs + pow(1 - cos_theta, 5.0) * (float3(1.0, 1.0, 1.0) - Rs);
    }
    
    void Init(float3 Rd_, float3 Rs_, Distribution distribution_, float3 normal_)
    {
        Rd = Rd_;
        Rs = Rs_;
        distribution = distribution_;
        frame.FromZ(normal_);
    }
    
    uint Flags()
    {
        return BSDF_GlossyReflection;
    }
    
    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (!SameHemisphere(wo, wi))
        {
            return 0.f;
        }
        
        float3 diffuse = (28.0 / (23.0 * PI)) * Rd * (1.0 - Rs) *
	               (1.0 - pow(1.0 - 0.5 * AbsCosTheta(wi), 5.0)) *
	               (1.0 - pow(1.0 - 0.5 * AbsCosTheta(wo), 5.0));

        float3 wm = wi + wo;
        if (wm.x == 0.0 && wm.y == 0.0 && wm.z == 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }
        wm = normalize(wm);
        
        float D = distribution.D(wm);
        float3 specular = D / (4.0 * abs(dot(wi, wm) * max(AbsCosTheta(wi), AbsCosTheta(wo)))) * SchlickFresnel(dot(wi, wm));

        return diffuse + specular;
    }
    
    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (!(flags & SampleFlags_Reflection) || !SameHemisphere(wo, wi))
        {
            return 0.f;
        }
        
        float3 wm = normalize(wo + wi);
        float pdf_wh = distribution.Pdf(wo, wm);

        return 0.5 * (AbsCosTheta(wi) * InvPI + pdf_wh / (4.0 * dot(wo, wm)));
    }
    
    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        
        BSDFSample bsdf_sample;
        
        bsdf_sample.Init();
        
        if (!(flags & SampleFlags_Reflection))
        {
            return bsdf_sample;
        }
        
        float3 wi = 0.f;
        float3 wm = 0.f;
        
        if (u.x < 0.5)
        {
            u.x = min(2.0 * u.x, 0.999999);
            wi = SampleCosineHemisphere(u);
            if (wo.z < 0.0)
            {
                wi.z *= -1.0;
            }
        }
        else
        {
            u.x = min(2.0 * (u.x - 0.5), 0.999999);
            wm = distribution.SampleWm(wo, u);
            wi = reflect(wo, wm);
            if (!SameHemisphere(wo, wi))
            {
                return bsdf_sample;
            }
        }
        
        float3 wiW = frame.ToWorld(wi);
        
        bsdf_sample.f = Eval(woW, wiW, mode);
        bsdf_sample.wiW = wiW;
        bsdf_sample.pdf = PDF(woW, wiW, mode, flags);
        bsdf_sample.flags = BSDF_GlossyReflection;
        bsdf_sample.eta = 1;
        
        return bsdf_sample;
    }
};

#endif