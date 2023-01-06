#ifndef BLEND_BSDF_HLSLI
#define BLEND_BSDF_HLSLI

#include "BSDF.hlsli"

template<typename T1, typename T2>
struct BlendBSDF
{
    T1 bsdf_1;
    T2 bsdf_2;
    float weight;

    void Init(T1 bsdf1, T2 bsdf2, float weight_)
    {
        bsdf_1 = bsdf1;
        bsdf_2 = bsdf2;
        weight = weight_;
    }

    float3 Eval(float3 wo, float3 wi, TransportMode mode)
    {
        return bsdf_1.Eval(wo, wi, mode) * weight + bsdf_2.Eval(wo, wi, mode) * (1.f - weight);
    }

    float PDF(float3 wo, float3 wi, TransportMode mode, SampleFlags flags)
    {
        float pdf1 = bsdf_1.PDF(wo, wi, mode, flags);
        float pdf2 = bsdf_1.PDF(wo, wi, mode, flags);
        
        if(pdf1 * pdf2 == 0)
        {
            if(pdf1 == 0)
            {
                return pdf1;
            }
            if(pdf2 == 0)
            {
                return pdf2;
            }
            return 0.0;
        }
        else
        {
            return (pdf1 + pdf2) * 0.5;
        }
    }

    BSDFSample Samplef(float3 wo, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        // TODO: Better random
        if((uc + u.x + u.y) * 0.33 < weight)
        {
            return bsdf_1.Samplef(wo, uc, u, mode, flags);
        }
        else
        {
            return bsdf_2.Samplef(wo, uc, u, mode, flags);
        }
    }
};

#endif