#ifndef LAYERED_BSDF_HLSLI
#define LAYERED_BSDF_HLSLI

#include "DiffuseBSDF.hlsli"

#define TopBxDF DiffuseBSDF
#define BottomBxDF DiffuseBSDF

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

    Frame frame;

    void Init()
    {

    }

    uint Flags()
    {

    }

    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {

    }

    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {

    }

    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {

    }
};

#endif