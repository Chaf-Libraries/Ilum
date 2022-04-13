#ifndef __MATERIAL_HLSL__
#define __MATERIAL_HLSL__

#include "BxDF.hlsli"
#include "Common.hlsli"

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
    BxDF bxdfs[7];
    uint nBxDFs;
    
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
#endif