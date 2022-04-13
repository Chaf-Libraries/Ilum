#include "../../RayTracing.hlsli"

#define USE_MicrofacetReflection
#define USE_LambertianReflection

#include "../../Material.hlsli"

BSDFs CreatePlasticMaterial(Interaction isect)
{
    BSDFs bsdfs;
    bsdfs.Init();
    bsdfs.isect = isect;
    
    float rough = max(isect.material.roughness, 0.001);
    float aspect = sqrt(1.0 - isect.material.anisotropic * 0.9);
    float urough = max(0.001, isect.material.roughness / aspect);
    float vrough = max(0.001, isect.material.roughness * aspect);

    lambertian_reflection.R = isect.material.base_color.rgb;

    microfacet_reflection.Distribution_Type = DistributionType_TrowbridgeReitz;
    microfacet_reflection.Fresnel_Type = FresnelType_Dielectric;
    microfacet_reflection.R = isect.material.data;
    microfacet_reflection.trowbridgereitz_distribution.alpha_x = rough;
    microfacet_reflection.trowbridgereitz_distribution.alpha_y = rough;
    microfacet_reflection.trowbridgereitz_distribution.sample_visible_area = true;
    microfacet_reflection.fresnel_dielectric.etaI = 1.0;
    microfacet_reflection.fresnel_dielectric.etaT = 1.5;
    
    bsdfs.AddBxDF(BxDF_MicrofacetReflection);
    bsdfs.AddBxDF(BxDF_LambertianReflection);
    
    return bsdfs;
}

[shader("callable")]
void main(inout BSDFSampleDesc bsdf)
{
    BSDFs mat = CreatePlasticMaterial(bsdf.isect);
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
