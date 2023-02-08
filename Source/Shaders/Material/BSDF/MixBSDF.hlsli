#ifndef MIX_BSDF_HLSLI
#define MIX_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../../Math.hlsli"
#include "../../Random.hlsli"
#include "../../Interaction.hlsli"

template<typename T1, typename T2>
struct MixBSDF
{
    T1 bsdf1;
    T2 bsdf2;
    float weight;

    void Init(T1 bsdf1_, T2 bsdf2_, float weight_)
    {
        bsdf1 = bsdf1_;
        bsdf2 = bsdf2_;
        weight = weight_;
    }

    uint Flags()
    {
        return bsdf1.Flags() | bsdf2.Flags();
    }

    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        if(weight <= 0)
        {
            return bsdf1.Eval(woW, wiW, mode);
        }
        else if(weight >= 1)
        {
            return bsdf2.Eval(woW, wiW, mode);
        }
        return lerp(bsdf1.Eval(woW, wiW, mode), bsdf2.Eval(woW, wiW, mode), weight);
        // PCGSampler rng;
        // rng.seed = HashCombine(Hash(woW), Hash(wiW));

        // if(rng.Get1D() < weight)
        // {
        //     return bsdf1.Eval(woW, wiW, mode);
        // }
        // else
        // {
        //     return bsdf2.Eval(woW, wiW, mode);
        // }
    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        if(weight <= 0)
        {
            return bsdf1.PDF(woW, wiW, mode, flags);
        }
        else if(weight >= 1)
        {
            return bsdf2.PDF(woW, wiW, mode, flags);
        }
        return lerp(bsdf1.PDF(woW, wiW, mode, flags), bsdf2.PDF(woW, wiW, mode, flags), weight);

        // PCGSampler rng;
        // rng.seed = HashCombine(Hash(woW), Hash(wiW));

        // if(rng.Get1D() < weight)
        // {
        //     return bsdf1.PDF(woW, wiW, mode, flags);
        // }
        // else
        // {
        //     return bsdf2.PDF(woW, wiW, mode, flags);
        // }
    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        if(weight <= 0)
        {
            return bsdf1.Samplef(woW, uc, u, mode, flags);
        }
        else if(weight >= 1)
        {
            return bsdf2.Samplef(woW, uc, u, mode, flags);
        }

        BSDFSample bsdf_sample;
        bsdf_sample.Init();

        if(uc < weight)
        {
            bsdf_sample = bsdf1.Samplef(woW, uc, u, mode, flags);
            bsdf_sample.pdf *= weight;
        }
        else
        {
            bsdf_sample = bsdf2.Samplef(woW, uc, u, mode, flags);
            bsdf_sample.pdf *= 1 - weight;
        }

        return bsdf_sample;
    }
};

#endif