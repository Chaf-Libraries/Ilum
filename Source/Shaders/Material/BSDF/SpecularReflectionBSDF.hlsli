#ifndef SPECULAR_REFLECTION_BSDF_HLSLI
#define SPECULAR_REFLECTION_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../../Math.hlsli"
#include "../../Interaction.hlsli"

template<typename Fresnel>
struct SpecularReflectionBSDF
{
    float3 R;
    Fresnel fresnel;
    Frame frame;
    
    void Init(float3 R_, Fresnel fresnel_, float3 normal_)
    {
        R = R_;
        fresnel = fresnel_;
        frame.FromZ(normal_);
    }
    
    uint Flags()
    {
        return BSDF_SpecularReflection;
    }
    
    float3 Eval(float3 woW, float3 wiW, TransportMode mode)
    {
        return float3(0.0, 0.0, 0.0);
    }
    
    float PDF(float3 woW, float3 wiW, TransportMode mode, SampleFlags flags)
    {
        return 0.0;
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
        
        float3 wi = float3(-wo.x, -wo.y, wo.z);
        float3 wiW = frame.ToWorld(wi);
        
        bsdf_sample.f = fresnel.Eval(CosTheta(wi)) * R / AbsCosTheta(wi);
        bsdf_sample.wiW = wiW;
        bsdf_sample.pdf = 1.f;
        bsdf_sample.flags = BSDF_DiffuseReflection;
        bsdf_sample.eta = 1;

        return bsdf_sample;
    }
};

#endif