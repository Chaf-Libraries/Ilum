#ifndef DIELECTRIC_BSDF_HLSLI
#define DIELECTRIC_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../Scattering.hlsli"
#include "../Fresnel.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct DielectricBSDF
{
    float3 R;
    float3 T;
    float eta;
    TrowbridgeReitzDistribution distribution;

    void Init(float3 R_, float3 T_, float ior_, float roughness)
    {
        R = R_;
        T = T_;
        eta = ior_;

        distribution.alpha_x = roughness;
        distribution.alpha_y = roughness;
        distribution.sample_visible_area = true;
    }

    uint Flags()
    {
        uint flags = (eta == 1.f? BSDF_Transmission : BSDF_Reflection | BSDF_Transmission);
        return flags | (distribution.EffectivelySmooth()? BSDF_Specular : BSDF_Glossy);
    }

    float3 Eval(float3 wo, float3 wi, TransportMode mode)
    {
        if(eta == 1.f || distribution.EffectivelySmooth())
        {
            return 0.f;
        }

        float cos_theta_o = CosTheta(wo);
        float cos_theta_i = CosTheta(wi);
        bool reflection = cos_theta_o * cos_theta_i > 0;
        float etap = 1;
        if(!reflection)
        {
            etap = cos_theta_o > 0.f ? eta : (1 / eta);
        }

        float3 wm = wi * etap + wo;

        if(cos_theta_i == 0 || cos_theta_o == 0 || LengthSquared(wm) == 0)
        {
            return 0.f;
        }

        wm = FaceForward(normalize(wm), float3(0, 0, 1));

        if(dot(wm, wi) * cos_theta_i < 0 || dot(wm, wo) * cos_theta_o < 0)
        {
            return 0.f;
        }

        float F = FresnelDielectric(dot(wo, wm), eta);

        if(reflection)
        {
            return R * distribution.D(wm) * distribution.G(wo, wi) * F / 
                abs(4 * cos_theta_i * cos_theta_o);            
        }
        else
        {
            float denom = Sqr(dot(wi, wm) + dot(wo, wm) / etap) * cos_theta_i * cos_theta_o;
            float3 ft = T * distribution.D(wm) * (1.f - F) * distribution.G(wo, wi) * 
                abs(dot(wi, wm) * dot(wo, wm) / denom);
            if(mode == TransportMode_Radiance)
            {
                ft /= Sqr(etap);
            }
            return ft;
        }
    }

    float PDF(float3 wo, float3 wi, TransportMode mode, SampleFlags flags)
    {
        if(eta == 1.f || distribution.EffectivelySmooth())
        {
            return 0.f;
        }

        float cos_theta_o = CosTheta(wo);
        float cos_theta_i = CosTheta(wi);
        bool reflection = cos_theta_o * cos_theta_i > 0;
        float etap = 1;
        if(!reflection)
        {
            etap = cos_theta_o > 0.f ? eta : 1 / eta;
        }

        float3 wm = wi * etap + wo;

        if(cos_theta_i == 0 || cos_theta_o == 0 || LengthSquared(wm) == 0)
        {
            return 0.f;
        }

        wm = FaceForward(normalize(wm), float3(0, 0, 1));

        if(dot(wm, wi) * cos_theta_i < 0 || dot(wm, wo) * cos_theta_o < 0)
        {
            return 0.f;
        }

        float pr = FresnelDielectric(dot(wo, wm), eta);
        float pt = 1.f - pr;

        if(!(flags & SampleFlags_Reflection))
        {
            pr = 0;
        }
        if(!(flags & SampleFlags_Transmission))
        {
            pt = 0;
        }
        if(pr == 0 && pt == 0)
        {
            return 0.f;
        }

        float pdf = 0.f;
        if(reflection)
        {
            pdf = distribution.Pdf(wo, wm) / (4.0 * abs(dot(wo, wm))) * pr / (pr + pt);
        }
        else
        {
            float denom = Sqr(dot(wi, wm) + dot(wo, wm) / etap);
            float dwm_dwi = abs(dot(wi, wm)) / denom;
            pdf = distribution.Pdf(wo, wm) * dwm_dwi * pt / (pr + pt);
        }

        return pdf;
    }

    BSDFSample Samplef(float3 wo, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        BSDFSample bsdf_sample;
        bsdf_sample.Init();

        if(eta == 1 || distribution.EffectivelySmooth())
        {
            // Smooth Dielectric
            float F = FresnelDielectric(CosTheta(wo), eta);
            float pr = F;
            float pt = 1.f - F;
            if(!(flags & SampleFlags_Reflection))
            {
                pr = 0.f;
            }
            if(!(flags & SampleFlags_Transmission))
            {
                pt = 0.f;
            }
            if(pr == 0.f && pt == 0.f)
            {
                return bsdf_sample;
            }

            if(uc < pr / (pr + pt))
            {
                float3 wi = float3(-wo.x, -wo.y, wo.z);
                float3 fr = R * F / abs(CosTheta(wi));
                
                bsdf_sample.f = fr;
                bsdf_sample.wi = wi;
                bsdf_sample.pdf = pr / (pr + pt);
                bsdf_sample.flags = BSDF_SpecularReflection;
                bsdf_sample.eta = 1;

                return bsdf_sample;
            }
            else
            {
                float etap = 0.f;
                float3 wi;
                if (!Refract(wo, float3(0.0, 0.0, 1.0), eta, etap, wi))
                {
                    return bsdf_sample;
                }

                float3 ft = (1.f - F) * T / abs(CosTheta(wi));
                if(mode == TransportMode_Radiance)
                {
                    ft /= Sqr(etap);
                }

                bsdf_sample.f = ft;
                bsdf_sample.wi = wi;
                bsdf_sample.pdf = pt / (pr + pt);
                bsdf_sample.flags = BSDF_SpecularTransmission;
                bsdf_sample.eta = etap;

                return bsdf_sample;
            }
        }
        else
        {
            // Rough Dielectric
            float3 wm = distribution.SampleWm(wo, u);
            float F = FresnelDielectric(CosTheta(wo), eta);
            float pr = F;
            float pt = 1.f - F;

            if(!(flags & SampleFlags_Reflection))
            {
                pr = 0.f;
            }
            if(!(flags & SampleFlags_Transmission))
            {
                pt = 0.f;
            }
            if(pr == 0.f && pt == 0.f)
            {
                return bsdf_sample;
            }

            if (uc < pr / (pr + pt))
            {
                float3 wi = Reflect(wo, wm);
                if(!SameHemisphere(wo, wi))
                {
                    return bsdf_sample;
                }

                bsdf_sample.f = distribution.D(wm) * distribution.G(wo, wi) * R * F / (4.f * CosTheta(wi) * CosTheta(wo));
                bsdf_sample.wi = wi;
                bsdf_sample.pdf = distribution.Pdf(wo, wm) / (4.f * abs(dot(wo, wm))) * pr / (pr + pt);
                bsdf_sample.flags = BSDF_GlossyReflection;
                bsdf_sample.eta = 1.f;

                return bsdf_sample;
            }
            else
            {
                float etap;
                float3 wi;
                if(!Refract(wo, Faceforward(wm, float3(0, 0, 1)), eta, etap, wi))
                {
                    return bsdf_sample;
                }

                float denom = Sqr(dot(wi, wm) + dot(wo, wm) / etap);
                float dwm_dwi = abs(dot(wi, wm)) / denom;
                float3 ft = T * (1.f - F) * distribution.D(wm) * distribution.G(wo, wi) * 
                        abs(dot(wi, wm) * dot(wo, wm) / 
                        (CosTheta(wi) * CosTheta(wo) * denom));

                if(mode == TransportMode_Radiance)
                {
                    ft /= Sqr(etap);
                }

                bsdf_sample.f = ft;
                bsdf_sample.wi = wi;
                bsdf_sample.pdf = distribution.Pdf(wo, wm) * dwm_dwi * pt / (pr + pt);
                bsdf_sample.flags = BSDF_GlossyTransmission;
                bsdf_sample.eta = etap;

                return bsdf_sample;
            }
        }

        return bsdf_sample;
    }
};

#endif