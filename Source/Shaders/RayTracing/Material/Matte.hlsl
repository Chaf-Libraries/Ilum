#define USE_OrenNayar
#define USE_LambertianReflection

#include "../../RayTracing.hlsli"

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
    bsdf.bsdf = CreateMatteMaterial(bsdf.isect);
    bsdf.eta = 1.0;
    
    if (bsdf.mode == BSDF_Evaluate)
    {
        bsdf.f = bsdf.bsdf.f(bsdf.woW, bsdf.wiW, bsdf.BxDF_Type);
    }
    else if (bsdf.mode == BSDF_Sample)
    {
        bsdf.f = bsdf.bsdf.Samplef(bsdf.woW, bsdf.rnd, bsdf.wiW, bsdf.pdf, bsdf.BxDF_Type, bsdf.sampled_type);
    }
    else if(bsdf.mode == BSDF_Pdf)
    {
        bsdf.pdf = bsdf.bsdf.Pdf(bsdf.woW, bsdf.wiW, bsdf.BxDF_Type);
    }
}
