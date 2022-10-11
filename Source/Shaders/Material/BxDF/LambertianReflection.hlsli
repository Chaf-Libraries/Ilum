#ifndef LAMBERTIANREFLECTION_HLSLI
#define LAMBERTIANREFLECTION_HLSLI

#include "../Utils/BxDFType.hlsli"
#include "../../Math.hlsli"

struct LambertianReflection
{
    float3 R;
    static const uint BxDF_Type = BxDF_REFLECTION | BxDF_DIFFUSE;
    
    void Init(float3 color)
    {
        R = color;
    }
    
    float3 Eval(float3 wi, float3 wo)
    {
        return R * InvPI;
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