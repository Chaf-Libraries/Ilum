#include "../../RayTracing.hlsli"

#define USE_OrenNayar
#define USE_LambertianReflection

#include "../../Material.hlsli"

BSDFs CreateMatteMaterial(Interaction isect)
{
    BSDFs bsdfs;
    bsdfs.Init();
    bsdfs.isect = isect;
    
    if (!IsBlack(isect.material.base_color.rgb))
    {
        if (isect.material.roughness == 0.0)
        {
            lambertian_reflection.R = isect.material.base_color.rgb;
            bsdfs.AddBxDF(BxDF_LambertianReflection);
        }
        else
        {
            oren_nayar.Init(isect.material.base_color.rgb, isect.material.roughness);
            bsdfs.AddBxDF(BxDF_OrenNayar);
        }
    }
    return bsdfs;
}

[shader("callable")]
void main(inout BSDFSampleDesc bsdf)
{
    BSDFs mat = CreateMatteMaterial(bsdf.isect);
    if (bsdf.mode == BSDF_Evaluate)
    {
        bsdf.f = mat.f(bsdf.woW, bsdf.wiW, bsdf.BxDF_Type);
    }
    else if (bsdf.mode == BSDF_Sample)
    {
        bsdf.f = mat.Samplef(bsdf.woW, bsdf.rnd, bsdf.wiW, bsdf.pdf, bsdf.BxDF_Type, bsdf.sampled_type);
    }
    else if(bsdf.mode == BSDF_Pdf)
    {
        bsdf.pdf = mat.Pdf(bsdf.woW, bsdf.wiW, bsdf.BxDF_Type);
    }
}
