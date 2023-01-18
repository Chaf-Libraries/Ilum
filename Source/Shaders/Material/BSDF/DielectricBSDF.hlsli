#ifndef DIELECTRIC_BSDF_HLSLI
#define DIELECTRIC_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../Scattering.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct DielectricBSDF
{
    TrowbridgeReitzDistribution distrib;
    float eta;
    Frame frame;

    void Init(float roughness_u_, float roughness_v_, float eta_, float3 normal_)
    {
        distrib.Init(distrib.RoughnessToAlpha(roughness_u_), distrib.RoughnessToAlpha(roughness_v_));
        eta = eta_;
        frame.FromZ(normal_);
    }
    
    uint Flags()
    {
        uint flags = (eta == 1) ? BSDF_Transmission :
            (BSDF_Reflection | BSDF_Transmission);
        return flags | (distrib.EffectivelySmooth() ? BSDF_Specular : BSDF_Glossy);

    }

    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (eta == 1 || distrib.EffectivelySmooth())
        {
            return 0.f;
        }
        
        // Evaluate rough dielectric BSDF
        float cosTheta_o = CosTheta(wo);
        float cosTheta_i = CosTheta(wi);
        bool reflect = cosTheta_i * cosTheta_o > 0;
        float etap = 1;
        
        if (!reflect)
        {
            etap = cosTheta_o > 0 ? eta : (1 / eta);
        }
        
        float3 wm = wi * etap + wo;
        if (cosTheta_i == 0 || cosTheta_o == 0 || LengthSquared(wm) == 0)
        {
            return 0.f;
        }
        
        wm = FaceForward(normalize(wm), float3(0, 0, 1));
        
        // Discard backfacing microfacets
        if (dot(wm, wi) * cosTheta_i < 0 || dot(wm, wo) * cosTheta_o < 0)
        {
            return 0.f;
        }
        
        float F = FresnelDielectric(dot(wo, wm), eta);
        if (reflect)
        {
             // Compute reflection at rough dielectric interface
            return distrib.D(wm) * distrib.G(wo, wi) * F / abs(4 * cosTheta_i * cosTheta_o);
        }
        else
        {
            // Compute transmission at rough dielectric interface
            float denom = Sqr(dot(wi, wm) + dot(wo, wm) / etap) * cosTheta_i * cosTheta_o;
            float ft = distrib.D(wm) * (1 - F) * distrib.G(wo, wi) * abs(dot(wi, wm) * dot(wo, wm) / denom);
            // Account for non-symmetry with transmission to different medium
            if (mode == TransportMode_Radiance)
            {
                ft /= Sqr(etap);
            }
            return ft;
        }
    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (eta == 1 || distrib.EffectivelySmooth())
        {
            return 0;
        }
        
        // Evaluate sampling PDF of rough dielectric BSDF
        // Compute generalized half vector _wm_
        float cosTheta_o = CosTheta(wo), cosTheta_i = CosTheta(wi);
        bool reflect = cosTheta_i * cosTheta_o > 0;
        float etap = 1;
        if (!reflect)
        {
            etap = cosTheta_o > 0 ? eta : (1 / eta);
        }
        float3 wm = wi * etap + wo;
        if (cosTheta_i == 0 || cosTheta_o == 0 || LengthSquared(wm) == 0)
        {
            return 0.f;
        }
        wm = FaceForward(normalize(wm), float3(0, 0, 1));

        // Discard backfacing microfacets
        if (dot(wm, wi) * cosTheta_i < 0 || dot(wm, wo) * cosTheta_o < 0)
        {
            return 0.f;
        }

        // Determine Fresnel reflectance of rough dielectric boundary
        float R = FresnelDielectric(dot(wo, wm), eta);
        float T = 1 - R;

        // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
        float pr = R, pt = T;
        if (!(flags & BSDF_Reflection))
        {
            pr = 0.f;
        }
        if (!(flags & BSDF_Transmission))
        {
            pt = 0.f;
        }
        if (pr == 0 && pt == 0)
        {
            return 0.f;
        }

        // Return PDF for rough dielectric
        float pdf;
        if (reflect)
        {
            // Compute PDF of rough dielectric reflection
            pdf = distrib.PDF(wo, wm) / (4 * abs(dot(wo, wm))) * pr / (pr + pt);
        }
        else
        {
            // Compute PDF of rough dielectric transmission
            float denom = Sqr(dot(wi, wm) + dot(wo, wm) / etap);
            float dwm_dwi = abs(dot(wi, wm)) / denom;
            pdf = distrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
        }
        return pdf;
    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        
        BSDFSample bsdf_sample;
        
        bsdf_sample.Init();
        
        if (eta == 1 || distrib.EffectivelySmooth())
        {
            // Sample perfect specular dielectric BSDF
            float R = FresnelDielectric(CosTheta(wo), eta), T = 1 - R;
            // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
            float pr = R, pt = T;
            if (!(flags & BSDF_Reflection))
            {
                pr = 0.f;
            }
            if (!(flags & BSDF_Transmission))
            {
                pt = 0.f;
            }
            if (pr == 0 && pt == 0)
            {
                return bsdf_sample;
            }
        
            if (uc < pr / (pr + pt))
            {
                // Sample perfect specular dielectric BRDF
                float3 wi = float3(-wo.x, -wo.y, wo.z);
                float3 fr = R / AbsCosTheta(wi);
                
                bsdf_sample.f = fr;
                bsdf_sample.wiW = frame.ToWorld(wi);
                bsdf_sample.pdf = pr / (pr + pt);
                bsdf_sample.flags = BSDF_SpecularReflection;
                return bsdf_sample;
            }
            else
            {
                // Sample perfect specular dielectric BTDF
                // Compute ray direction for specular transmission
                float3 wi;
                float etap;
                if (!Refract(wo, float3(0, 0, 1), eta, etap, wi))
                {
                    return bsdf_sample;
                }
                
                float3 ft = T / AbsCosTheta(wi);
                // Account for non-symmetry with transmission to different medium
                if (mode == TransportMode_Radiance)
                {
                    ft /= Sqr(etap);
                }
                
                bsdf_sample.f = ft;
                bsdf_sample.wiW = frame.ToWorld(wi);
                bsdf_sample.pdf = pt / (pr + pt);
                bsdf_sample.flags = BSDF_SpecularTransmission;
                bsdf_sample.eta = etap;
                return bsdf_sample;
            }
        }
        else
        {
            // Sample rough dielectric BSDF
            float3 wm = distrib.Sample_wm(Faceforward(wo, float3(0, 0, 1)), u);
            float R = FresnelDielectric(dot(wo, wm), eta);
            float T = 1 - R;
            // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
            float pr = R, pt = T;
            if (!(flags & BSDF_Reflection))
            {
                pr = 0.f;
            }
            if (!(flags & BSDF_Transmission))
            {
                pt = 0.f;
            }
            if (pr == 0 && pt == 0)
            {
                return bsdf_sample;
            }

            if (uc < pr / (pr + pt))
            {
                // Sample reflection at rough dielectric interface
                float3 wi = Reflect(wo, wm);
                if (!SameHemisphere(wo, wi))
                {
                    return bsdf_sample;
                }
            
                bsdf_sample.f = distrib.D(wm) * distrib.G(wo, wi) * R / (4 * CosTheta(wi) * CosTheta(wo));
                bsdf_sample.wiW = frame.ToWorld(wi);
                bsdf_sample.pdf = distrib.PDF(wo, wm) / (4 * abs(dot(wo, wm))) * pr / (pr + pt);
                bsdf_sample.flags = BSDF_GlossyReflection;
                return bsdf_sample;
            }
            else
            {
               // Sample transmission at rough dielectric interface
                float etap;
                float3 wi;
                bool tir = !Refract(wo, wm, eta, etap, wi);
                if (SameHemisphere(wo, wi) || wi.z == 0 || tir)
                {
                    return bsdf_sample;
                }
               // Compute PDF of rough dielectric transmission
                float denom = Sqr(dot(wi, wm) + dot(wo, wm) / etap);
                float dwm_dwi = abs(dot(wi, wm)) / denom;
            
               // Evaluate BRDF and return _BSDFSample_ for rough transmission
                float3 ft = T * distrib.D(wm) * distrib.G(wo, wi) * abs(dot(wi, wm) * dot(wo, wm) / (CosTheta(wi) * CosTheta(wo) * denom));
               // Account for non-symmetry with transmission to different medium
                if (mode == TransportMode_Radiance)
                {
                    ft /= Sqr(etap);
                }
               
                bsdf_sample.f = ft;
                bsdf_sample.wiW = frame.ToWorld(wi);
                bsdf_sample.pdf = distrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
                bsdf_sample.flags = BSDF_GlossyTransmission;
                bsdf_sample.eta = etap;
                return bsdf_sample;
            }
        }
       
        return bsdf_sample;
    }
};

#endif