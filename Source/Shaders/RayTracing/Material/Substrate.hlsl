#include "../../RayTracing.hlsli"

#define USE_FresnelBlend

#include "../../Material.hlsli"

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
    BSDFs mat = CreateSubstrateMaterial(bsdf.isect);
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
