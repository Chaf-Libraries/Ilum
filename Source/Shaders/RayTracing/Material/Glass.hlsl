#include "../../RayTracing.hlsli"

#define USE_FresnelSpecular
#define USE_MicrofacetReflection
#define USE_MicrofacetTransmission

#include "../../Material.hlsli"

BSDFs CreateGlassMaterial(Interaction isect)
{
    BSDFs bsdfs;
    bsdfs.Init();
    bsdfs.isect = isect;
    
    float aspect = sqrt(1.0 - isect.material.anisotropic * 0.9);
    float urough = isect.material.roughness / aspect;
    float vrough = isect.material.roughness * aspect;
    
    float3 R = isect.material.base_color.rgb;
    float3 T = isect.material.data;
    
    float refraction = isect.material.refraction;
    float anisotropic = isect.material.anisotropic;
    float roughness = isect.material.roughness;
        
    if (IsBlack(R) && IsBlack(T))
    {
        return bsdfs;
    }
    
    bool isSpecular = (urough == 0 && vrough == 0);
    
    if (isSpecular)
    {
        bsdfs.AddBxDF(BxDF_FresnelSpecular);
    }
    else
    {
        if (!IsBlack(R))
        {
            bsdfs.AddBxDF(BxDF_MicrofacetReflection);
        }
        if (!IsBlack(T))
        {
            bsdfs.AddBxDF(BxDF_MicrofacetTransmission);
        }
    }
            
    fresnel_specular.R = R;
    fresnel_specular.T = T;
    fresnel_specular.etaA = 1.0;
    fresnel_specular.etaB = refraction;
    fresnel_specular.mode = TransportMode_Radiance;
        
    microfacet_reflection.Distribution_Type = DistributionType_TrowbridgeReitz;
    microfacet_reflection.Fresnel_Type = FresnelType_Dielectric;
    microfacet_reflection.R = R;
    microfacet_reflection.fresnel_dielectric.etaI = 1.0;
    microfacet_reflection.fresnel_dielectric.etaT = refraction;
    microfacet_reflection.trowbridgereitz_distribution.alpha_x = urough;
    microfacet_reflection.trowbridgereitz_distribution.alpha_y = vrough;
    microfacet_reflection.trowbridgereitz_distribution.sample_visible_area = true;
        
    microfacet_transmission.Distribution_Type = DistributionType_TrowbridgeReitz;
    microfacet_transmission.Fresnel_Type = FresnelType_Dielectric;
    microfacet_transmission.T = T;
    microfacet_transmission.etaA = 1.0;
    microfacet_transmission.etaB = refraction;
    microfacet_transmission.fresnel_dielectric = microfacet_reflection.fresnel_dielectric;
    microfacet_transmission.trowbridgereitz_distribution = microfacet_reflection.trowbridgereitz_distribution;
    
    return bsdfs;
}

[shader("callable")]
void main(inout BSDFSampleDesc bsdf)
{
    BSDFs mat = CreateGlassMaterial(bsdf.isect);
    if (bsdf.mode == BSDF_Evaluate)
    {
        bsdf.f = mat.f(bsdf.woW, bsdf.wiW, bsdf.BxDF_Type);
    }
    else if (bsdf.mode == BSDF_Sample)
    {
        bsdf.f = mat.Samplef(bsdf.woW, bsdf.rnd, bsdf.wiW, bsdf.pdf, bsdf.BxDF_Type, bsdf.sampled_type);
    }
    else if (bsdf.mode == BSDF_Pdf)
    {
        bsdf.pdf = mat.Pdf(bsdf.woW, bsdf.wiW, bsdf.BxDF_Type);
    }
}
