#define USE_MicrofacetReflection

#include "../../RayTracing.hlsli"

BSDFs CreateMetalMaterial(Interaction isect)
{
    BSDFs bsdfs;
    bsdfs.Init();
    bsdfs.isect = isect;
    
    bsdfs.AddBxDF(BxDF_MicrofacetReflection);
    
    float aspect = sqrt(1.0 - isect.material.anisotropic * 0.9);
    float urough = max(0.001, isect.material.roughness / aspect);
    float vrough = max(0.001, isect.material.roughness * aspect);
        
    microfacet_reflection.Fresnel_Type = FresnelType_Conductor;
    microfacet_reflection.Distribution_Type = DistributionType_TrowbridgeReitz;
    microfacet_reflection.R = isect.material.base_color.rgb;
    microfacet_reflection.fresnel_conductor.etaI = float3(1.0, 1.0, 1.0);
    microfacet_reflection.fresnel_conductor.etaT = float3(1, 10, 11);
    microfacet_reflection.fresnel_conductor.k = float3(3.90463543, 2.44763327, 2.13765264);
    microfacet_reflection.trowbridgereitz_distribution.alpha_x = urough;
    microfacet_reflection.trowbridgereitz_distribution.alpha_y = vrough;
    microfacet_reflection.trowbridgereitz_distribution.sample_visible_area = true;
    
    return bsdfs;
}

[shader("callable")]
void main(inout BSDFSampleDesc bsdf)
{
    bsdf.bsdf = CreateMetalMaterial(bsdf.isect);
    bsdf.eta = 1.0;
    
    if (bsdf.mode == BSDF_Evaluate)
    {
        bsdf.f = bsdf.bsdf.f(bsdf.woW, bsdf.wiW, bsdf.BxDF_Type);
    }
    else if (bsdf.mode == BSDF_Sample)
    {
        bsdf.f = bsdf.bsdf.Samplef(bsdf.woW, bsdf.rnd, bsdf.wiW, bsdf.pdf, bsdf.BxDF_Type, bsdf.sampled_type);
    }
    else if (bsdf.mode == BSDF_Pdf)
    {
        bsdf.pdf = bsdf.bsdf.Pdf(bsdf.woW, bsdf.wiW, bsdf.BxDF_Type);
    }
}
