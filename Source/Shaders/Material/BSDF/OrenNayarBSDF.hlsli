#ifndef OREN_NAYAR_BSDF_HLSLI
#define OREN_NAYAR_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct OrenNayarBSDF
{
    float3 R;
    float A, B;
    Frame frame;
    
    void Init(float3 R_, float sigma, float3 normal_)
    {
        R = R_;
        sigma = Radians(sigma);
        float sigma2 = sigma * sigma;
        A = 1.0 - sigma2 / (2.0 * (sigma2 + 0.33));
        B = 0.45 * sigma2 / (sigma2 + 0.09);
        frame.FromZ(normal_);
    }
    
    uint Flags()
    {
        return BSDF_DiffuseReflection;
    }
    
    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (!SameHemisphere(wo, wi))
        {
            return 0.f;
        }
        
        float sin_theta_i = SinTheta(wi);
        float sin_theta_o = SinTheta(wo);

        float max_cos = 0;
        if (sin_theta_i > 1e-4 && sin_theta_o > 1e-4)
        {
            float sin_phi_i = SinPhi(wi);
            float cos_phi_i = CosPhi(wi);
            float sin_phi_o = SinPhi(wo);
            float cos_phi_o = CosPhi(wo);

            float d_cos = cos_phi_i * cos_phi_o + sin_phi_i * sin_phi_o;
            max_cos = max(0.0, d_cos);
        }

        float tan_theta_i = sin_theta_i / AbsCosTheta(wi);
        float tan_theta_o = sin_theta_o / AbsCosTheta(wo);

        float sin_alpha, tan_beta;
        if (AbsCosTheta(wi) > AbsCosTheta(wo))
        {
            sin_alpha = sin_theta_o;
            tan_beta = sin_theta_i / AbsCosTheta(wi);
        }
        else
        {
            sin_alpha = sin_theta_i;
            tan_beta = sin_theta_o / AbsCosTheta(wo);
        }
        
        return R * InvPI * (A + B * max_cos * sin_alpha * tan_beta);
    }
    
    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        float3 wo = frame.ToLocal(woW);
        float3 wi = frame.ToLocal(wiW);
        
        if (!(flags & SampleFlags_Reflection) || !SameHemisphere(wo, wi))
        {
            return 0.f;
        }
        
        return AbsCosTheta(wi) * InvPI;
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

        float3 wi = SampleCosineHemisphere(u);
        
        if (wo.z < 0.f)
        {
            wi.z *= -1.f;
        }

        float3 wiW = frame.ToWorld(wi);
        
        bsdf_sample.f = Eval(woW, wiW, mode);
        bsdf_sample.wiW = wiW;
        bsdf_sample.pdf = PDF(woW, wiW, mode, flags);
        bsdf_sample.flags = BSDF_DiffuseReflection;
        bsdf_sample.eta = 1;

        return bsdf_sample;
    }
};

#endif