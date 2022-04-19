#ifndef __MATERIAL_HLSL__
#define __MATERIAL_HLSL__

#include "BxDF.hlsli"
#include "Common.hlsli"

//#define USE_Matte
//#define USE_Plastic
//#define USE_Metal
//#define USE_Substrate
//#define USE_Mirror
//#define USE_Glass
//#define USE_Disney

#ifdef USE_Matte
#define USE_OrenNayar
#define USE_LambertianReflection
#endif

#ifdef USE_Plastic
#define USE_MicrofacetReflection
#define USE_LambertianReflection
#endif

#ifdef USE_Metal
#define USE_MicrofacetReflection
#endif

#ifdef USE_Substrate
#define USE_FresnelBlend
#endif

#ifdef USE_Mirror
#define USE_SpecularReflection
#endif

#ifdef USE_Glass
#define USE_FresnelSpecular
#define USE_MicrofacetReflection
#define USE_MicrofacetTransmission
#endif

#ifdef USE_Disney
#define USE_DisneyDiffuse
#define USE_DisneyFakeSS
#define USE_SpecularTransmission
#define USE_DisneyRetro
#define USE_DisneySheen
#define USE_DisneyClearcoat
#define USE_MicrofacetReflection
#define USE_MicrofacetTransmission
#define USE_LambertianTransmission
#endif

#ifdef USE_OrenNayar
static OrenNayar oren_nayar;
#endif

#ifdef USE_LambertianReflection
static LambertianReflection lambertian_reflection;
#endif

#ifdef USE_MicrofacetReflection
static MicrofacetReflection microfacet_reflection;
#endif

#ifdef USE_SpecularReflection
static SpecularReflection specular_reflection;
#endif

#ifdef USE_SpecularTransmission
static SpecularTransmission specular_transmission;
#endif

#ifdef USE_FresnelBlend
static FresnelBlend fresnel_blend;
#endif

#ifdef USE_FresnelSpecular
static FresnelSpecular fresnel_specular;
#endif

#ifdef USE_MicrofacetTransmission
static MicrofacetTransmission microfacet_transmission;
#endif

#ifdef USE_DisneyDiffuse
static DisneyDiffuse disney_diffuse;
#endif

#ifdef USE_DisneyFakeSS
static DisneyFakeSS disney_fakess;
#endif

#ifdef USE_DisneyRetro
static DisneyRetro disney_retro;
#endif

#ifdef USE_DisneySheen
static DisneySheen disney_sheen;
#endif

#ifdef USE_DisneyClearcoat
static DisneyClearcoat disney_clearcoat;
#endif

#ifdef USE_LambertianTransmission
static LambertianTransmission lambertian_transmission;
#endif

////////////// BxDF //////////////
struct BxDF
{
    uint BxDF_Flag;
        
    float3 f(float3 wo, float3 wi)
    {
        switch (BxDF_Flag)
        {
#ifdef USE_OrenNayar
            case BxDF_OrenNayar:
                return oren_nayar.f(wo, wi);
#endif

#ifdef USE_LambertianReflection
            case BxDF_LambertianReflection:
                return lambertian_reflection.f(wo, wi);
#endif

#ifdef USE_MicrofacetReflection
            case BxDF_MicrofacetReflection:
                return microfacet_reflection.f(wo, wi);
#endif
            
#ifdef USE_SpecularReflection
            case BxDF_SpecularReflection:
                return specular_reflection.f(wo, wi);
#endif
            
#ifdef USE_SpecularTransmission
            case BxDF_SpecularTransmission:
                return specular_transmission.f(wo, wi);
#endif
            
#ifdef USE_FresnelBlend
            case BxDF_FresnelBlend:
                return fresnel_blend.f(wo, wi);
#endif
            
#ifdef USE_FresnelSpecular
            case BxDF_FresnelSpecular:
                return fresnel_specular.f(wo, wi);
#endif
            
#ifdef USE_MicrofacetTransmission
            case BxDF_MicrofacetTransmission:
                return microfacet_transmission.f(wo, wi);
#endif
            
#ifdef USE_DisneyDiffuse
            case BxDF_DisneyDiffuse:
                return disney_diffuse.f(wo, wi);
#endif
            
#ifdef USE_DisneyFakeSS
            case BxDF_DisneyFakeSS:
                return disney_fakess.f(wo, wi);
#endif
            
#ifdef USE_DisneyRetro
            case BxDF_DisneyRetro:
                return disney_retro.f(wo, wi);
#endif
            
#ifdef USE_DisneySheen
            case BxDF_DisneySheen:
                return disney_sheen.f(wo, wi);
#endif
            
#ifdef USE_DisneyClearcoat
            case BxDF_DisneyClearcoat:
                return disney_clearcoat.f(wo, wi);
#endif
            
#ifdef USE_LambertianTransmission
            case BxDF_LambertianTransmission:
                return lambertian_transmission.f(wo, wi);
#endif
        }
        return float3(0.0, 0.0, 0.0);
    }
    
    float Pdf(float3 wo, float3 wi)
    {
        switch (BxDF_Flag)
        {
#ifdef USE_OrenNayar
            case BxDF_OrenNayar:
                return oren_nayar.Pdf(wo, wi);
#endif
            
#ifdef USE_LambertianReflection
            case BxDF_LambertianReflection:
                return lambertian_reflection.Pdf(wo, wi);
#endif
            
#ifdef USE_MicrofacetReflection
            case BxDF_MicrofacetReflection:
                return microfacet_reflection.Pdf(wo, wi);
#endif
            
#ifdef USE_SpecularReflection
            case BxDF_SpecularReflection:
                return specular_reflection.Pdf(wo, wi);
#endif
            
#ifdef USE_SpecularTransmission
            case BxDF_SpecularTransmission:
                return specular_transmission.Pdf(wo, wi);
#endif
            
#ifdef USE_FresnelBlend
            case BxDF_FresnelBlend:
                return fresnel_blend.Pdf(wo, wi);
#endif
            
#ifdef USE_FresnelSpecular
            case BxDF_FresnelSpecular:
                return fresnel_specular.Pdf(wo, wi);
#endif
            
#ifdef USE_MicrofacetTransmission
            case BxDF_MicrofacetTransmission:
                return microfacet_transmission.Pdf(wo, wi);
#endif
            
#ifdef USE_DisneyDiffuse
            case BxDF_DisneyDiffuse:
                return disney_diffuse.Pdf(wo, wi);
#endif
            
#ifdef USE_DisneyFakeSS
            case BxDF_DisneyFakeSS:
                return disney_fakess.Pdf(wo, wi);
#endif
            
#ifdef USE_DisneyRetro
            case BxDF_DisneyRetro:
                return disney_retro.Pdf(wo, wi);
#endif
            
#ifdef USE_DisneySheen
            case BxDF_DisneySheen:
                return disney_sheen.Pdf(wo, wi);
#endif
            
#ifdef USE_DisneyClearcoat
            case BxDF_DisneyClearcoat:
                return disney_clearcoat.Pdf(wo, wi);
#endif
            
#ifdef USE_LambertianTransmission
            case BxDF_LambertianTransmission:
                return lambertian_transmission.Pdf(wo, wi);
#endif
        }
        return 0.0;
    }
    
    float3 Samplef(float3 wo, float2 u, out float3 wi, out float pdf)
    {
        switch (BxDF_Flag)
        {
#ifdef USE_OrenNayar
            case BxDF_OrenNayar:
                return oren_nayar.Samplef(wo, u, wi, pdf);
#endif
            
#ifdef USE_LambertianReflection
            case BxDF_LambertianReflection:
                return lambertian_reflection.Samplef(wo, u, wi, pdf);
#endif
            
#ifdef USE_MicrofacetReflection
            case BxDF_MicrofacetReflection:
                return microfacet_reflection.Samplef(wo, u, wi, pdf);
#endif
            
#ifdef USE_SpecularReflection
            case BxDF_SpecularReflection:
                return specular_reflection.Samplef(wo, u, wi, pdf);
#endif
            
#ifdef USE_SpecularTransmission
            case BxDF_SpecularTransmission:
                return specular_transmission.Samplef(wo, u, wi, pdf);
#endif
            
#ifdef USE_FresnelBlend
            case BxDF_FresnelBlend:
                return fresnel_blend.Samplef(wo, u, wi, pdf);
#endif
            
#ifdef USE_FresnelSpecular
            case BxDF_FresnelSpecular:
                return fresnel_specular.Samplef(wo, u, wi, pdf);
#endif
            
#ifdef USE_MicrofacetTransmission
            case BxDF_MicrofacetTransmission:
                return microfacet_transmission.Samplef(wo, u, wi, pdf);
#endif
            
#ifdef USE_DisneyDiffuse
            case BxDF_DisneyDiffuse:
                return disney_diffuse.Samplef(wo, u, wi, pdf);
#endif
            
#ifdef USE_DisneyFakeSS
            case BxDF_DisneyFakeSS:
                return disney_fakess.Samplef(wo, u, wi, pdf);
#endif
            
#ifdef USE_DisneyRetro
            case BxDF_DisneyRetro:
                return disney_retro.Samplef(wo, u, wi, pdf);
#endif
            
#ifdef USE_DisneySheen
            case BxDF_DisneySheen:
                return disney_sheen.Samplef(wo, u, wi, pdf);
#endif
            
#ifdef USE_DisneyClearcoat
            case BxDF_DisneyClearcoat:
                return disney_clearcoat.Samplef(wo, u, wi, pdf);
#endif
            
#ifdef USE_LambertianTransmission
            case BxDF_LambertianTransmission:
                return lambertian_transmission.Samplef(wo, u, wi, pdf);
#endif
        }
        return float3(0.0, 0.0, 0.0);
    }
    
    uint GetBxDFType()
    {
        switch (BxDF_Flag)
        {
#ifdef USE_OrenNayar
            case BxDF_OrenNayar:
                return oren_nayar.BxDF_Type;
#endif
            
#ifdef USE_LambertianReflection
            case BxDF_LambertianReflection:
                return lambertian_reflection.BxDF_Type;
#endif
            
#ifdef USE_MicrofacetReflection
            case BxDF_MicrofacetReflection:
                return microfacet_reflection.BxDF_Type;
#endif
            
#ifdef USE_SpecularReflection
            case BxDF_SpecularReflection:
                return specular_reflection.BxDF_Type;
#endif
            
#ifdef USE_SpecularTransmission
            case BxDF_SpecularTransmission:
                return specular_transmission.BxDF_Type;
#endif
            
#ifdef USE_FresnelBlend
            case BxDF_FresnelBlend:
                return fresnel_blend.BxDF_Type;
#endif
            
#ifdef USE_FresnelSpecular
            case BxDF_FresnelSpecular:
                return fresnel_specular.BxDF_Type;
#endif
            
#ifdef USE_MicrofacetTransmission
            case BxDF_MicrofacetTransmission:
                return microfacet_transmission.BxDF_Type;
#endif
            
#ifdef USE_DisneyDiffuse
            case BxDF_DisneyDiffuse:
                return disney_diffuse.BxDF_Type;
#endif
            
#ifdef USE_DisneyFakeSS
            case BxDF_DisneyFakeSS:
                return disney_fakess.BxDF_Type;
#endif
            
#ifdef USE_DisneyRetro
            case BxDF_DisneyRetro:
                return disney_retro.BxDF_Type;
#endif
            
#ifdef USE_DisneySheen
            case BxDF_DisneySheen:
                return disney_sheen.BxDF_Type;
#endif
            
#ifdef USE_DisneyClearcoat
            case BxDF_DisneyClearcoat:
                return disney_clearcoat.BxDF_Type;
#endif
            
#ifdef USE_LambertianTransmission
            case BxDF_LambertianTransmission:
                return lambertian_transmission.BxDF_Type;
#endif
        }
        return 0;
    }
    
    bool MatchesType(uint type)
    {
        return (GetBxDFType() & type) == GetBxDFType();
    }
};

////////////// BSDFs //////////////
struct BSDFs
{
    uint BxDF_Flags;
    BxDF bxdfs[10];
    uint nBxDFs;
    
    float eta;
    
    Interaction isect;
    
    void Init()
    {
        BxDF_Flags = 0;
        nBxDFs = 0;
    }
    
    void AddBxDF(uint flag)
    {
        BxDF_Flags |= flag;
        bxdfs[nBxDFs].BxDF_Flag = flag;
        nBxDFs++;
    }
    
    int NumComponents(uint BxDF_type)
    {
        int num = 0;
        for (int i = 0; i < nBxDFs; i++)
        {
            if (bxdfs[i].MatchesType(BxDF_type))
            {
                num++;
            }
        }
        return num;
    }
    
    uint GetComponents()
    {
        uint cmpts = 0;
        for (int i = 0; i < nBxDFs; i++)
        {
            cmpts |= bxdfs[i].GetBxDFType();
        }
        return cmpts;
    }
    
    float3 f(float3 woW, float3 wiW, uint BxDF_type)
    {
        float3 wi = isect.WorldToLocal(wiW);
        float3 wo = isect.WorldToLocal(woW);

        if (wo.z == 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }
        bool reflect = dot(wiW, isect.ffnormal) * dot(woW, isect.ffnormal) > 0;
        float3 f = { 0.0, 0.0, 0.0 };
        for (uint i = 0; i < nBxDFs; i++)
        {
            if (bxdfs[i].MatchesType(BxDF_type) &&
                ((reflect && (bxdfs[i].GetBxDFType() & BSDF_REFLECTION)) ||
                (!reflect && (bxdfs[i].GetBxDFType() & BSDF_TRANSMISSION))))
            {
                f += bxdfs[i].f(wo, wi);
            }
        }
        return f;
    }
    
    float Pdf(float3 woW, float3 wiW, uint BxDF_type)
    {
        if (nBxDFs == 0)
        {
            return 0.0;
        }
        float3 wi = isect.WorldToLocal(wiW);
        float3 wo = isect.WorldToLocal(woW);
        if (wo.z == 0.0)
        {
            return 0.0;
        }
        float pdf = 0.0;
        int matching_cmpts = 0;
        for (uint i = 0; i < nBxDFs; i++)
        {
            if (bxdfs[i].MatchesType(BxDF_type))
            {
                matching_cmpts++;
                pdf += bxdfs[i].Pdf(wo, wi);
            }
        }
        return matching_cmpts > 0 ? pdf / matching_cmpts : 0.0;
    }
    
    float3 Samplef(float3 woW, float2 u, out float3 wiW, out float pdf, uint BxDF_type, out uint sampled_type)
    {
        int matching_compts = NumComponents(BxDF_type);
        if (matching_compts == 0)
        {
            pdf = 0.0;
            sampled_type = 0.0;
            return float3(0.0, 0.0, 0.0);
        }
        int comp = min((int) (u.x * matching_compts), matching_compts - 1);

        uint bxdf_idx = 0;
        int count = comp;
        for (int i = 0; i < nBxDFs; i++)
        {
            if (bxdfs[i].MatchesType(BxDF_type) && count-- == 0)
            {
                bxdf_idx = i;
                break;
            }
        }
        
        float2 uRemapped = float2(min(u.x * matching_compts - comp, 0.99999999999999), u.y);

        BxDF bxdf = bxdfs[bxdf_idx];
                
        float3 wi = { 0.0, 0.0, 0.0 };
        float3 wo = isect.WorldToLocal(woW);
        if (wo.z == 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }
        
        pdf = 0.0;
        sampled_type = bxdf.GetBxDFType();
        
        float3 f = bxdf.Samplef(wo, uRemapped, wi, pdf);

        if (pdf == 0.0)
        {
            sampled_type = 0;
            return float3(0.0, 0.0, 0.0);
        }
        
        wiW = isect.LocalToWorld(wi);
        
        if (!(bxdf.GetBxDFType() & BSDF_SPECULAR) && matching_compts > 1)
        {
            for (int i = 0; i < nBxDFs; i++)
            {
                if (i != bxdf_idx && bxdfs[i].MatchesType(BxDF_type))
                {
                    pdf += bxdfs[i].Pdf(wo, wi);
                }
            }
        }
        
        if (matching_compts > 1)
        {
            pdf /= matching_compts;
        }

        if (!(bxdf.GetBxDFType() & BSDF_SPECULAR))
        {
            bool reflect = dot(wiW, isect.ffnormal) * dot(woW, isect.ffnormal) > 0;
            f = 0;
            for (int i = 0; i < nBxDFs; i++)
            {
                if (bxdfs[i].MatchesType(BxDF_type) &&
                    ((reflect && (bxdfs[i].GetBxDFType() & BSDF_REFLECTION)) ||
                    (!reflect && (bxdfs[i].GetBxDFType() & BSDF_TRANSMISSION))))
                {
                    f += bxdfs[i].f(wo, wi);
                }
            }
        }

        return f;
    }
};

#ifdef USE_Matte
BSDFs CreateMatteMaterial(Material material)
{
    BSDFs bsdfs;
    bsdfs.Init();
    
    bsdfs.eta = 1.0;
    
    if (!IsBlack(material.base_color.rgb))
    {
        if (material.roughness == 0.0)
        {
            lambertian_reflection.R = material.base_color.rgb;
            bsdfs.AddBxDF(BxDF_LambertianReflection);
        }
        else
        {
            oren_nayar.Init(material.base_color.rgb, material.roughness);
            bsdfs.AddBxDF(BxDF_OrenNayar);
        }
    }
    return bsdfs;
}
#endif

#ifdef USE_Plastic
BSDFs CreatePlasticMaterial(Material material)
{
    BSDFs bsdfs;
    bsdfs.Init();

    bsdfs.eta = 1.0;
    
    float rough = max(material.roughness, 0.001);
    float aspect = sqrt(1.0 - material.anisotropic * 0.9);
    float urough = max(0.001, material.roughness / aspect);
    float vrough = max(0.001, material.roughness * aspect);

    lambertian_reflection.R = material.base_color.rgb;

    microfacet_reflection.Distribution_Type = DistributionType_TrowbridgeReitz;
    microfacet_reflection.Fresnel_Type = FresnelType_Dielectric;
    microfacet_reflection.R = material.data;
    microfacet_reflection.trowbridgereitz_distribution.alpha_x = rough;
    microfacet_reflection.trowbridgereitz_distribution.alpha_y = rough;
    microfacet_reflection.trowbridgereitz_distribution.sample_visible_area = true;
    microfacet_reflection.fresnel_dielectric.etaI = 1.0;
    microfacet_reflection.fresnel_dielectric.etaT = 1.5;
    
    bsdfs.AddBxDF(BxDF_MicrofacetReflection);
    bsdfs.AddBxDF(BxDF_LambertianReflection);
    
    return bsdfs;
}
#endif

#ifdef USE_Metal
BSDFs CreateMetalMaterial(Material material)
{
    BSDFs bsdfs;
    bsdfs.Init();

    bsdfs.eta = 1.0;
    
    bsdfs.AddBxDF(BxDF_MicrofacetReflection);
    
    float aspect = sqrt(1.0 - material.anisotropic * 0.9);
    float urough = max(0.001, material.roughness / aspect);
    float vrough = max(0.001, material.roughness * aspect);
        
    microfacet_reflection.Fresnel_Type = FresnelType_Conductor;
    microfacet_reflection.Distribution_Type = DistributionType_TrowbridgeReitz;
    microfacet_reflection.R = material.base_color.rgb;
    microfacet_reflection.fresnel_conductor.etaI = float3(1.0, 1.0, 1.0);
    microfacet_reflection.fresnel_conductor.etaT = float3(1, 10, 11);
    microfacet_reflection.fresnel_conductor.k = float3(3.90463543, 2.44763327, 2.13765264);
    microfacet_reflection.trowbridgereitz_distribution.alpha_x = urough;
    microfacet_reflection.trowbridgereitz_distribution.alpha_y = vrough;
    microfacet_reflection.trowbridgereitz_distribution.sample_visible_area = true;
    
    return bsdfs;
}
#endif

#ifdef USE_Substrate
BSDFs CreateSubstrateMaterial(Material material)
{
    BSDFs bsdfs;
    bsdfs.Init();
    
    bsdfs.eta = 1.0;
    
    bsdfs.AddBxDF(BxDF_FresnelBlend);
    
    float aspect = sqrt(1.0 - material.anisotropic * 0.9);
    float urough = max(0.001, material.roughness / aspect);
    float vrough = max(0.001, material.roughness * aspect);
    
    fresnel_blend.Distribution_Type = DistributionType_TrowbridgeReitz;
    fresnel_blend.Rd = material.base_color.rgb;
    fresnel_blend.Rs = material.data;
    fresnel_blend.trowbridgereitz_distribution.alpha_x = urough;
    fresnel_blend.trowbridgereitz_distribution.alpha_y = vrough;
    fresnel_blend.trowbridgereitz_distribution.sample_visible_area = true;
    
    return bsdfs;
}
#endif

#ifdef USE_Mirror
BSDFs CreateMirrorMaterial(Material material)
{
    BSDFs bsdfs;
    bsdfs.Init();
    bsdfs.eta = 1.0;
    bsdfs.AddBxDF(BxDF_SpecularReflection);
    specular_reflection.R = material.base_color.rgb;
    return bsdfs;
}
#endif

#ifdef USE_Glass
BSDFs CreateGlassMaterial(Material material)
{
    BSDFs bsdfs;
    bsdfs.Init();
    
    bsdfs.eta = material.refraction;

    float aspect = sqrt(1.0 - material.anisotropic * 0.9);
    float urough = material.roughness / aspect;
    float vrough = material.roughness * aspect;
    
    float3 R = material.base_color.rgb;
    float3 T = material.data;
    
    float refraction = material.refraction;
    float anisotropic = material.anisotropic;
    float roughness = material.roughness;
        
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
    microfacet_transmission.T = T;
    microfacet_transmission.etaA = 1.0;
    microfacet_transmission.etaB = refraction;
    microfacet_transmission.trowbridgereitz_distribution = microfacet_reflection.trowbridgereitz_distribution;
    
    return bsdfs;
}
#endif

#ifdef USE_Disney
BSDFs CreateDisneyMaterial(Material material)
{
    BSDFs bsdfs;
    bsdfs.Init();

    bsdfs.eta = 1.0;
    
    float3 c = material.base_color.rgb;
    float metallicWeight = material.metallic;
    float e = material.refraction;
    float strans = material.specular_transmission;
    float diffuseWeight = (1.0 - metallicWeight) * (1.0 - strans);
    float dt = material.diffuse_transmission * 0.5;
    float rough = material.roughness;
    float lum = dot(float3(0.212671, 0.715160, 0.072169), c);
    float3 Ctint = lum > 0.0 ? (c / lum) : float3(1.0, 1.0, 1.0);
    float sheenWeight = material.sheen;
    float3 Csheen = float3(0.0, 0.0, 0.0);
    if (sheenWeight > 0.0)
    {
        float stint = material.sheen_tint;
        Csheen = lerp(float3(1.0, 1.0, 1.0), Ctint, stint);
    }
    
    if (diffuseWeight > 0.0)
    {
        if (material.thin > 0)
        {
            float flat = material.flatness;
            // Add Disney Diffuse
            bsdfs.AddBxDF(BxDF_DisneyDiffuse);
            disney_diffuse.R = diffuseWeight * (1.0 - flat) * (1.0 - dt) * c;
            
            // Add Disney FakeSS
            bsdfs.AddBxDF(BxDF_DisneyFakeSS);
            disney_fakess.R = diffuseWeight * flat * (1.0 - dt) * c;
            disney_fakess.roughness = rough;
        }
        else
        {
            float3 scatter_distance = material.data;
            if (IsBlack(scatter_distance))
            {
                // No subsurface scattering
                // Add Disney Diffuse
                bsdfs.AddBxDF(BxDF_DisneyDiffuse);
                disney_diffuse.R = diffuseWeight * c;
            }
            else
            {
                // Use BSSRDF
                // Add Specular Transmission
                bsdfs.AddBxDF(BxDF_SpecularTransmission);
                specular_transmission.T = float3(1.0, 1.0, 1.0);
                specular_transmission.etaA = 1.0;
                specular_transmission.etaB = e;
                specular_transmission.fresnel.etaI = specular_transmission.etaA;
                specular_transmission.fresnel.etaT = specular_transmission.etaB;
                specular_transmission.mode = TransportMode_Radiance;
                                
                // TODO: Add Disney BSSRDF
            }
        }
        
        // Add Disney Retro Reflection
        bsdfs.AddBxDF(BxDF_DisneyRetro);
        disney_retro.R = diffuseWeight * c;
        disney_retro.roughness = rough;
        
        // Sheen
        if (sheenWeight > 0.0)
        {
            // Add Disney Sheen
            bsdfs.AddBxDF(BxDF_DisneySheen);
            disney_sheen.R = diffuseWeight * sheenWeight * Csheen;
        }
    }
    
    float aspect = sqrt(1.0 - material.anisotropic * 0.9);
    float ax = max(0.001, sqrt(rough) / aspect);
    float ay = max(0.001, sqrt(rough) * aspect);
    
    float specTint = material.specular_tint;
    float3 Cspec0 = lerp(SchlickR0FromEta(e) * lerp(float3(1.0, 1.0, 1.0), Ctint, specTint), c, metallicWeight);
    
    // Add Microfacet Reflection
    bsdfs.AddBxDF(BxDF_MicrofacetReflection);
    microfacet_reflection.R = float3(1.0, 1.0, 1.0);
    microfacet_reflection.Fresnel_Type = FresnelType_Disney;
    microfacet_reflection.fresnel_disney.R0 = Cspec0;
    microfacet_reflection.fresnel_disney.metallic = metallicWeight;
    microfacet_reflection.fresnel_disney.eta = e;
    microfacet_reflection.Distribution_Type = DistributionType_Disney;
    microfacet_reflection.disney_distribution.alpha_x = ax;
    microfacet_reflection.disney_distribution.alpha_y = ay;
    microfacet_reflection.disney_distribution.sample_visible_area = true;
    
    // Clearcoat
    float cc = material.clearcoat;
    if (cc > 0.0)
    {
        // Add Disney Clear Coat
        bsdfs.AddBxDF(BxDF_DisneyClearcoat);
        disney_clearcoat.weight = cc;
        disney_clearcoat.gloss = lerp(0.1, 0.001, material.clearcoat_gloss);
    }
    
    // BTDF
    if (strans > 0.0)
    {
        float3 T = strans * sqrt(c);
        if (material.thin > 0.0)
        {
            float rscaled = (0.65 * e - 0.35) * rough;
            float ax = max(0.001, rscaled * rscaled / aspect);
            float ay = max(0.001, rscaled * rscaled * aspect);
            // Add Microfacet Transmission with GGX
            bsdfs.AddBxDF(BxDF_MicrofacetTransmission);
            microfacet_transmission.Distribution_Type = DistributionType_TrowbridgeReitz;
            microfacet_transmission.etaA = 1.0;
            microfacet_transmission.etaB = e;
            microfacet_transmission.T = T;
            microfacet_transmission.trowbridgereitz_distribution.alpha_x = ax;
            microfacet_transmission.trowbridgereitz_distribution.alpha_y = ay;
            microfacet_transmission.trowbridgereitz_distribution.sample_visible_area = true;
            microfacet_transmission.mode = TransportMode_Radiance;
        }
        else
        {
            // Add Microfacet Transmission with Disney Distribution
            bsdfs.AddBxDF(BxDF_MicrofacetTransmission);
            microfacet_transmission.Distribution_Type = DistributionType_Disney;
            microfacet_transmission.etaA = 1.0;
            microfacet_transmission.etaB = e;
            microfacet_transmission.T = T;
            microfacet_transmission.disney_distribution.alpha_x = ax;
            microfacet_transmission.disney_distribution.alpha_y = ay;
            microfacet_transmission.disney_distribution.sample_visible_area = true;
            microfacet_transmission.mode = TransportMode_Radiance;
        }
    }
    if (material.thin > 0.0)
    {
        // Add Lambertian Transmission
        bsdfs.AddBxDF(BxDF_LambertianTransmission);
        lambertian_transmission.T = dt * c;
    }
    
    return bsdfs;
}
#endif

#endif