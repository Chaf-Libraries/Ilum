#ifndef MIRROR_BSDF_HLSLI
#define MIRROR_BSDF_HLSLI

#include "BSDF.hlsli"
#include "SpecularReflectionBSDF.hlsli"
#include "../Fresnel.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

struct MirrorBSDF
{
    SpecularReflectionBSDF<FresnelOp> specular_reflection;
    
    void Init(float3 R_, float3 normal_)
    {
        FresnelOp fresnel;
        specular_reflection.Init(R_, fresnel, normal_);
    }
    
    uint Flags()
    {
        return BSDF_SpecularReflection;
    }
    
    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        return specular_reflection.Eval(woW, wiW, mode);
    }
    
    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        return specular_reflection.PDF(woW, wiW, mode, flags);
    }
    
    BSDFSample Samplef(float3 woW, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        return specular_reflection.Samplef(woW, uc, u, mode, flags);
    }
};

#endif