#ifndef LAYERED_BSDF_HLSLI
#define LAYERED_BSDF_HLSLI

#include "DiffuseBSDF.hlsli"
#include "../../Random.hlsli"

#define TopBxDF DiffuseBSDF
#define BottomBxDF DiffuseBSDF

// template<typename TopBxDF, typename BottomBxDF>
struct TopOrBottomBSDF
{
    TopBxDF top;
    BottomBxDF bottom;
    bool has_top;

    void SetTop(TopBxDF top_)
    {
        top = top_;
        has_top = true;
    }

    void SetBottom(BottomBxDF bottom_)
    {
        bottom = bottom_;
        has_top = false;
    }

    float3 Eval(float3 wo, float3 wi, TransportMode mode)
    {
        return has_top ? top.Eval(wo, wi, mode) : bottom.Eval(wo, wi, mode);
    }

    float PDF(float3 wo, float3 wi, TransportMode mode, SampleFlags flags)
    {
        return has_top ? top.PDF(wo, wi, mode, flags) : bottom.PDF(wo, wi, mode, flags);
    }

    BSDFSample Samplef(float3 wo, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        return has_top ? top.Samplef(wo, uc, u, mode, flags) : bottom.Samplef(wo, uc, u, mode, flags);
    }
};

// template<typename TopBxDF, typename BottomBxDF>
struct LayeredBSDF
{
    TopBxDF top;
    BottomBxDF bottom;
    bool two_side;

    float thickness;
    uint max_depth;
    float3 albedo;
    float g;
    uint sample_count;

    void Init()
    {

    }

    uint Flags()
    {
        uint top_flags = top.Flags();
        uint bottom_flags = bottom.Flags();
        uint flags = BSDF_Reflection;
 
        if(IsSpecular(top_flags))
        {
            flags |= BSDF_Specular;
        }

        if(IsDiffuse(top_flags) || IsDiffuse(bottom_flags) || !IsBlack(albedo))
        {
            flags |= BSDF_Glossy;
        }

        if(IsTransmissive(top_flags) && IsTransmissive(bottom_flags))
        {
            flags |= BSDF_Transmission;
        }

        return flags;
    }

    float3 Eval(float3 wo, float3 wi, TransportMode mode)
    {
        float3 f = 0.f;
        if(two_side && wo.z < 0)
        {
            wo = -wo;
            wi = -wi;
        }

        // Entrance interface
        TopOrBottomBSDF enter_interface;
        bool entered_top = two_side || wo.z > 0;
        if(entered_top)
        {
            enter_interface.SetTop(top);
        }
        else
        {
            enter_interface.SetBottom(bottom);
        }

        // Exit interface and exit Z
        TopOrBottomBSDF exit_interface, non_exit_interface;
        if(SameHemisphere(wo, wi) ^ entered_top)
        {
            exit_interface.SetBottom(bottom);
            non_exit_interface.SetTop(top);
        }
        else
        {
            exit_interface.SetTop(top);
            non_exit_interface.SetBottom(bottom);
        }

        float exit_z = (SameHemisphere(wo, wi) ^ entered_top) ? 0 : thickness;

        // Setup RNG
        PCGSampler rng;
        rng.seed = HashCombine(Hash(wo), Hash(wi));

        for(int s = 0; s < sample_count; s++)
        {
            // Random walk
            float uc = rng.Get1D();
            BSDFSample wos = enter_interface.Samplef(wo, uc, float2(rng.Get1D(), rng.Get1D()), mode, SampleFlags_Transmission);
            if(wos.pdf == 0 || IsBlack(wos.f) || wos.wi || wos.wi.z == 0)
            {
                continue;
            }

            uc = rng.Get1D();
            BSDFSample wis = enter_interface.Samplef(wi, uc, float2(rng.Get1D(), rng.Get1D()), !mode, SampleFlags_Transmission);
            if(wis.pdf == 0 || IsBlack(wis.f) || wis.wi || wis.wi.z == 0)
            {
                continue;
            }

            float3 beta = wos.f * AbsCosTheta(wos.wi) / wos.pdf;
            float z = entered_top ? thickness : 0;
            float3 w = wos.wi;
        }

    }

    float PDF(float3 wo, float3 wi, TransportMode mode, SampleFlags flags)
    {

    }

    BSDFSample Samplef(float3 wo, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {

    }
};

#endif