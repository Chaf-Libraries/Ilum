#ifndef __MATERIAL_HLSL__
#define __MATERIAL_HLSL__

#include "BxDF.hlsli"
#include "Common.hlsli"

static OrenNayar oren_nayar;
static LambertianReflection lambertian_reflection;
static MicrofacetReflection microfacet_reflection;
static SpecularReflection specular_reflection;
static SpecularTransmission specular_transmission;
static FresnelBlend fresnel_blend;
static FresnelSpecular fresnel_specular;
static MicrofacetTransmission microfacet_transmission;
static DisneyDiffuse disney_diffuse;

////////////// BxDF //////////////
struct BxDF
{
    uint BxDF_Flag;
        
    float3 f(float3 wo, float3 wi)
    {
        switch (BxDF_Flag)
        {
            case BxDF_OrenNayar:
                return oren_nayar.f(wo, wi);
            case BxDF_LambertianReflection:
                return lambertian_reflection.f(wo, wi);
            case BxDF_MicrofacetReflection:
                return microfacet_reflection.f(wo, wi);
            case BxDF_SpecularReflection:
                return specular_reflection.f(wo, wi);
            case BxDF_SpecularTransmission:
                return specular_transmission.f(wo, wi);
            case BxDF_FresnelBlend:
                return fresnel_blend.f(wo, wi);
            case BxDF_FresnelSpecular:
                return fresnel_specular.f(wo, wi);
            case BxDF_MicrofacetTransmission:
                return microfacet_transmission.f(wo, wi);
            case BxDF_DisneyDiffuse:
                return disney_diffuse.f(wo, wi);
        }
        return float3(0.0, 0.0, 0.0);
    }
    
    float Pdf(float3 wo, float3 wi)
    {
        switch (BxDF_Flag)
        {
            case BxDF_OrenNayar:
                return oren_nayar.Pdf(wo, wi);
            case BxDF_LambertianReflection:
                return lambertian_reflection.Pdf(wo, wi);
            case BxDF_MicrofacetReflection:
                return microfacet_reflection.Pdf(wo, wi);
            case BxDF_SpecularReflection:
                return specular_reflection.Pdf(wo, wi);
            case BxDF_SpecularTransmission:
                return specular_transmission.Pdf(wo, wi);
            case BxDF_FresnelBlend:
                return fresnel_blend.Pdf(wo, wi);
            case BxDF_FresnelSpecular:
                return fresnel_specular.Pdf(wo, wi);
            case BxDF_MicrofacetTransmission:
                return microfacet_transmission.Pdf(wo, wi);
            case BxDF_DisneyDiffuse:
                return disney_diffuse.Pdf(wo, wi);
        }
        return 0.0;
    }
    
    float3 Samplef(float3 wo, float2 u, out float3 wi, out float pdf)
    {
        switch (BxDF_Flag)
        {
            case BxDF_OrenNayar:
                return oren_nayar.Samplef(wo, u, wi, pdf);
            case BxDF_LambertianReflection:
                return lambertian_reflection.Samplef(wo, u, wi, pdf);
            case BxDF_MicrofacetReflection:
                return microfacet_reflection.Samplef(wo, u, wi, pdf);
            case BxDF_SpecularReflection:
                return specular_reflection.Samplef(wo, u, wi, pdf);
            case BxDF_SpecularTransmission:
                return specular_transmission.Samplef(wo, u, wi, pdf);
            case BxDF_FresnelBlend:
                return fresnel_blend.Samplef(wo, u, wi, pdf);
            case BxDF_FresnelSpecular:
                return fresnel_specular.Samplef(wo, u, wi, pdf);
            case BxDF_MicrofacetTransmission:
                return microfacet_transmission.Samplef(wo, u, wi, pdf);
            case BxDF_DisneyDiffuse:
                return disney_diffuse.Samplef(wo, u, wi, pdf);
        }
        return float3(0.0, 0.0, 0.0);
    }
    
    uint GetBxDFType()
    {
        switch (BxDF_Flag)
        {
            case BxDF_OrenNayar:
                return oren_nayar.BxDF_Type;
            case BxDF_LambertianReflection:
                return lambertian_reflection.BxDF_Type;
            case BxDF_MicrofacetReflection:
                return microfacet_reflection.BxDF_Type;
            case BxDF_SpecularReflection:
                return specular_reflection.BxDF_Type;
            case BxDF_SpecularTransmission:
                return specular_transmission.BxDF_Type;
            case BxDF_FresnelBlend:
                return fresnel_blend.BxDF_Type;
            case BxDF_FresnelSpecular:
                return fresnel_specular.BxDF_Type;
            case BxDF_MicrofacetTransmission:
                return microfacet_transmission.BxDF_Type;
            case BxDF_DisneyDiffuse:
                return disney_diffuse.BxDF_Type;
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
    BxDF bxdfs[5];
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
    
    float3 Samplef(float3 woW, Sampler _sampler, out float3 wiW, out float pdf, uint BxDF_type, out uint sampled_type)
    {
        float2 u = _sampler.Get2D();
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
                        
        BxDF bxdf = bxdfs[bxdf_idx];
                
        float3 wi = { 0.0, 0.0, 0.0 };
        float3 wo = isect.WorldToLocal(woW);
        if (wo.z == 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }
        
        pdf = 0.0;
        sampled_type = bxdf.GetBxDFType();
        float3 f = bxdf.Samplef(wo, _sampler.Get2D(), wi, pdf);

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

        return f;;
    }
};

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

BSDFs CreateMirrorMaterial(Interaction isect)
{
    BSDFs bsdfs;
    bsdfs.Init();
    bsdfs.isect = isect;
    bsdfs.AddBxDF(BxDF_SpecularReflection);
    specular_reflection.R = isect.material.base_color.rgb;
    return bsdfs;
}

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

BSDFs CreateDisneyMaterial(Interaction isect)
{
    BSDFs bsdfs;
    bsdfs.Init();
    bsdfs.isect = isect;
    
    float3 c = isect.material.base_color.rgb;
    float metallicWeight = isect.material.metallic;
    float e = isect.material.refraction;
    float strans = isect.material.specular_transmission;
    float diffuseWeight = (1.0 - metallicWeight) * (1.0 - strans);
    float dt = isect.material.diffuse_transmission * 0.5;
    float rough = isect.material.roughness;
    float lum = c.y;
    float3 Ctint = lum > 0.0 ? (c / lum) : float3(1.0, 1.0, 1.0);
    float sheenWeight = isect.material.sheen;
    float3 Csheen = float3(0.0, 0.0, 0.0);
    if (sheenWeight > 0.0)
    {
        float stint = isect.material.sheen_tint;
        Csheen = lerp(float3(1.0, 1.0, 1.0), Ctint, stint);
    }
    
    if(diffuseWeight>0.0)
    {
        if(isect.material.thin>0)
        {
            float flat = isect.material.flatness;
            // Add Disney Diffuse
            bsdfs.AddBxDF(BxDF_DisneyDiffuse);
            disney_diffuse.R = diffuseWeight * (1.0 - flat) * (1.0 - dt) * c;
            
            // TODO: Add Disney FakeSS
        }
        else
        {
            float3 scatter_distance = isect.material.data;
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
                
                // TODO: Add Disney BSSRDF
            }
        }
        
        // TODO: Add Disney Retro Reflection
        
        // Sheen
        if(sheenWeight>0.0)
        {
            // TODO: Add Disney Sheen
        }
    }
    
    float aspect = sqrt(1.0 - isect.material.anisotropic * 0.9);
    float ax = max(0.001, sqrt(rough) / aspect);
    float ay = max(0.001, sqrt(rough) * aspect);
    // TODO: Init Disney Microfacet Distribution
    
    float specTint = isect.material.specular_tint;
    float3 Cspec0 = lerp(SchlickR0FromEta(e) * lerp(float3(1.0, 1.0, 1.0), Ctint, specTint), c, metallicWeight);
    // TODO: Init Diseny Fresnel
    // TODO: Add Microfacet Reflection
    
    // Clearcoat
    float cc = isect.material.clearcoat;
    if(cc>0.0)
    {
        // TODO: Add Disney Clear Coat
    }
    
    // BTDF
    if(strans>0.0)
    {
        float3 T = strans * sqrt(c);
        if(isect.material.thin>0.0)
        {
            float rscaled = (0.65 * e - 0.35) * rough;
            float ax = max(0.001, rscaled * rscaled / aspect);
            float ay = max(0.001, rscaled * rscaled * aspect);
            // Add Microfacet Transmission with GGX
        }
        else
        {
            // TODO: Add Microfacet Transmission with Disney Distribution
        }
    }
    if(isect.material.thin>0.0)
    {
        // TODO: Add Lambertian Transmission
    }
    
    return bsdfs;
}

////////////// BSDF //////////////
struct BSDF
{
    Material mat;
    Interaction isect;
    BSDFs bsdf;
    
    void Init(Interaction isect_)
    {
        mat = isect_.material;
        isect = isect_;
        
        switch (mat.material_type)
        {
            case Material_Matte:
                bsdf = CreateMatteMaterial(isect);
                break;
            case Material_Plastic:
                bsdf = CreatePlasticMaterial(isect);
                break;
            case Material_Mirror:
                bsdf = CreateMirrorMaterial(isect);
                break;
            case Material_Metal:
                bsdf = CreateMetalMaterial(isect);
                break;
            case Material_Substrate:
                bsdf = CreateSubstrateMaterial(isect);
                break;
            case Material_Glass:
                bsdf = CreateGlassMaterial(isect);
                break;
            case Material_Disney:
                bsdf = CreateDisneyMaterial(isect);
                break;
        }
    }
    
    float3 f(float3 wo, float3 wi)
    {
        return bsdf.f(wo, wi, BSDF_ALL);
    }
    
    float3 Samplef(float3 wo, Sampler _sampler, out float3 wi, out float pdf)
    {
        uint type;
        return bsdf.Samplef(wo, _sampler, wi, pdf, BSDF_ALL, type);
    }
};

#endif