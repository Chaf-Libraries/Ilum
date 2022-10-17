#ifndef DIFFUSE_BSDF_HLSLI
#define DIFFUSE_BSDF_HLSLI

#include "../../Math.hlsli"

struct LambertianReflection
{
    static float3 Eval(float3 R, float3 wo, float3 wi)
    {
        return R * InvPI;
    }
    
    static float Pdf(float3 wo, float3 wi)
    {
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPI : 0.0;
    }
    
    static float3 Samplef(float3 R, float3 wo, float uc, float2 u, out float3 wi, out float pdf)
    {
        wi = UniformSampleHemisphere(u);
        if (wo.z < 0.0)
        {
            wi.z *= -1.0;
        }
        pdf = Pdf(wo, wi);
        return Eval(R, wo, wi);
    }
};

struct OrenNayarReflection
{
    static float3 Eval(float3 R, float sigma, float3 wo, float3 wi)
    {
        sigma = Radians(sigma);
        float sigma2 = sigma * sigma;
        float A = 1.0 - sigma2 / (2.0 * (sigma2 + 0.33));
        float B = 0.45 * sigma2 / (sigma2 + 0.09);
        
        float sinThetaI = SinTheta(wi);
        float sinThetaO = SinTheta(wo);

        float maxCos = 0;
        if (sinThetaI > 1e-4 && sinThetaO > 1e-4)
        {
            float sinPhiI = SinPhi(wi);
            float cosPhiI = CosPhi(wi);
            float sinPhiO = SinPhi(wo);
            float cosPhiO = CosPhi(wo);

            float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
            maxCos = max(0.0, dCos);
        }

        float tanThetaI = sinThetaI / AbsCosTheta(wi);
        float tanThetaO = sinThetaO / AbsCosTheta(wo);

        float sinAlpha, tanBeta;
        if (AbsCosTheta(wi) > AbsCosTheta(wo))
        {
            sinAlpha = sinThetaO;
            tanBeta = sinThetaI / AbsCosTheta(wi);
        }
        else
        {
            sinAlpha = sinThetaI;
            tanBeta = sinThetaO / AbsCosTheta(wo);
        }
        return R * InvPI * (A + B * maxCos * sinAlpha * tanBeta);
    }
    
    static float Pdf(float3 wo, float3 wi)
    {
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPI : 0.0;
    }
    
    static float3 Samplef(float3 R, float sigma, float3 wo, float uc, float2 u, out float3 wi, out float pdf)
    {
        wi = SampleCosineHemisphere(u);
        if (wo.z < 0.0)
        {
            wi.z *= -1.0;
        }
        pdf = Pdf(wo, wi);
        return Eval(R, sigma, wo, wi);
    }
};

struct DiffuseBSDF
{
    static float3 Eval(float3 R, float sigma, float3 wo, float3 wi)
    {
        sigma == 0.f ? LambertianReflection::Eval(R, wo, wi) : OrenNayarReflection::Eval(R, sigma, wo, wi);
    }

    static float3 Pdf(float3 R, float sigma, float3 wo, float3 wi)
    {
        sigma == 0.f ? LambertianReflection::Pdf(wo, wi) : OrenNayarReflection::Pdf(wo, wi);
    }

    static float3 Samplef(float3 R, float sigma, float3 wo, float uc, float2 u, out float3 wi, out float pdf)
    {
        sigma == 0.f ? LambertianReflection::Samplef(R, wo, uc, u, wi, pdf) : OrenNayarReflection::Samplef(R, sigma, wo, u, wi, pdf);
    }
}

#endif