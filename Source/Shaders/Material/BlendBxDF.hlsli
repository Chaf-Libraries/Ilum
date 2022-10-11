
#include "../BSDF.hlsli"

struct BlendBxDF
{
    float weight;
    
    BSDF A;
    BSDF B;
    
    void Init(float weight_)
    {
        weight = weight_;
    }
    
    float3 Eval(float3 V, float3 L)
    {
        return A.Eval(V, L) * (1.f - weight) + B.Eval(V, L) * weight;
    }
    
    float Pdf(float3 V, float3 L)
    {
        return A.Pdf(V, L) * (1.f - weight) + B.Pdf(V, L) * weight;
    }
    
    float3 Samplef(float V, float sample1, float2 sample2, out float3 L, out float pdf)
    {
        return 0.f;
    }
};
