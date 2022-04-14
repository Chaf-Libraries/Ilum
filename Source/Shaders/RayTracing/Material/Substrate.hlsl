#define USE_FresnelBlend

#include "../../RayTracing.hlsli"

BSDFs CreateSubstrateMaterial(Interaction isect)
{
    BSDFs bsdfs;
    bsdfs.Init();
    bsdfs.isect = isect;
    
    bsdfs.AddBxDF(BxDF_FresnelBlend);
    
    float aspect = sqrt(1.0 - isect.material.anisotropic * 0.9);
    float urough = max(0.001, isect.material.roughness / aspect);
    float vrough = max(0.001, isect.material.roughness * aspect);
    
    fresnel_blend.Distribution_Type = DistributionType_TrowbridgeReitz;
    fresnel_blend.Rd = isect.material.base_color.rgb;
    fresnel_blend.Rs = isect.material.data;
    fresnel_blend.trowbridgereitz_distribution.alpha_x = urough;
    fresnel_blend.trowbridgereitz_distribution.alpha_y = vrough;
    fresnel_blend.trowbridgereitz_distribution.sample_visible_area = true;
    
    return bsdfs;
}

[shader("callable")]
void main(inout BSDFSampleDesc bsdf)
{
    bsdf.bsdf = CreateSubstrateMaterial(bsdf.isect);
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
