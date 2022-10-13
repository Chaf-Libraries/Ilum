#ifndef LAMBERTIAN_REFLECTION_HLSLI
#define LAMBERTIAN_REFLECTION_HLSLI

#include "../BxDFType.hlsli"
#include "../../Math.hlsli"

struct LambertianReflection
{
    float3 BaseColor;
    static const uint BxDF_Type = BxDF_REFLECTION | BxDF_DIFFUSE;
    
    static LambertianReflection Create(float3 color)
    {
        LambertianReflection bxdf;
        bxdf.BaseColor = color;
        return bxdf;
    }
     
    float3 Eval(float3 wi, float3 wo)
    {
        return BaseColor * InvPI;
    }
    
    float Pdf(float3 wi, float3 wo)
    {
        return SameHemisphere(wi, wo) ? AbsCosTheta(wi) * InvPI : 0.0;
    }
    
    float3 Samplef(float3 wi, float sample1, float2 sample2, out float3 wo, out float pdf)
    {
        wi = UniformSampleHemisphere(sample2);
        if (wo.z < 0.0)
        {
            wi.z *= -1.0;
        }
        pdf = Pdf(wo, wi);
        return Eval(wo, wi);
    }
};
#endif