#include "../../RayTracing.hlsli"

#define USE_SpecularReflection

#include "../../Material.hlsli"

BSDFs CreateMirrorMaterial(Interaction isect)
{
    BSDFs bsdfs;
    bsdfs.Init();
    bsdfs.isect = isect;
    bsdfs.AddBxDF(BxDF_SpecularReflection);
    specular_reflection.R = isect.material.base_color.rgb;
    return bsdfs;
}

[shader("callable")]
void main(inout BSDFSampleDesc bsdf)
{
    BSDFs mat = CreateMirrorMaterial(bsdf.isect);
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
