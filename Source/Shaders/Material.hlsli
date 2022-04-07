#ifndef __MATERIAL_HLSL__
#define __MATERIAL_HLSL__

#include "BxDF.hlsli"
#include "Common.hlsli"

////////////// Matte Material //////////////
struct MatteMaterial
{
    float3 Kd;
    float sigma;
    
    LambertianReflection lambertian;
    OrenNayar oren_nayar;
    
    void Init(Material mat)
    {
        Kd = mat.base_color.rgb;
        sigma = mat.roughness;
        
        lambertian.R = Kd;
        oren_nayar.Init(Kd, sigma);
    }
    
    float3 f(float3 wo, float3 wi)
    {
        if (wo.z == 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }

        if (sigma == 0)
        {
            return lambertian.f(wo, wi);
        }
        else
        {
            return oren_nayar.f(wo, wi);
        }
    }
    
    float3 Samplef(float3 wo, Sampler _sampler, out float3 wi, out float pdf)
    {
        if (wo.z == 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }

        if (sigma == 0)
        {
            return lambertian.Samplef(wo, _sampler, wi, pdf);
        }
        else
        {
            return oren_nayar.Samplef(wo, _sampler, wi, pdf);
        }
    }
};

////////////// Plastic Material //////////////
struct PlasticMaterial
{
    float3 Kd;
    float3 Ks;
    float roughness;
    
    LambertianReflection lambertian;
    MicrofacetReflection microfacet;
    
    void Init(Material mat)
    {
        Kd = mat.base_color.rgb;
        Ks = mat.data;
        roughness = mat.roughness;
        
        lambertian.R = Kd;
        
        float rough = max(mat.roughness, 0.001);
        
        microfacet.Distribution_Type = DistributionType_TrowbridgeReitz;
        microfacet.Fresnel_Type = FresnelType_Dielectric;
        microfacet.R = Ks;
        microfacet.trowbridgereitz_distribution.alpha_x = rough;
        microfacet.trowbridgereitz_distribution.alpha_y = rough;
        microfacet.trowbridgereitz_distribution.sample_visible_area = true;
        microfacet.fresnel_dielectric.etaI = 1.0;
        microfacet.fresnel_dielectric.etaT = 1.5;
    }
    
    float3 f(float3 wo, float3 wi)
    {
        float3 distribution = float3(0.0, 0.0, 0.0);

	    // Diffuse term
        if (!IsBlack(Kd))
        {
            distribution += lambertian.f(wo, wi);
        }

	    // Specular term
        if (!IsBlack(Ks))
        {
            distribution += microfacet.f(wo, wi);
        }

        return distribution;
    }
    
    float3 Samplef(float3 wo, Sampler _sampler, out float3 wi, out float pdf)
    {
        float3 distribution = float3(0.0, 0.0, 0.0);

	// Random Select one bxdf to sample
        float u = _sampler.Get1D();

        if (u < 0.5)
        {
		    // Choose Lambertian term
            return lambertian.Samplef(wo, _sampler, wi, pdf);
        }
        else
        {
		    // Choose microfacet term
            return microfacet.Samplef(wo, _sampler, wi, pdf);
        }
        
        return float3(0.0, 0.0, 0.0);
    }
};

////////////// Metal Material //////////////
struct MetalMaterial
{
    float3 R;
    float3 eta;
    float3 k;
    float anisotropic;
    float roughness;
    
    MicrofacetReflection microfacet;
    
    void Init(Material mat)
    {
        R = mat.base_color.rgb;
        eta = float3(1, 10, 11);
        k = float3(3.90463543, 2.44763327, 2.13765264);
        anisotropic = mat.anisotropic;
        roughness = mat.roughness;
        
        float aspect = sqrt(1.0 - anisotropic * 0.9);
        float urough = max(0.001, roughness / aspect);
        float vrough = max(0.001, roughness * aspect);
        
        microfacet.Fresnel_Type = FresnelType_Conductor;
        microfacet.Distribution_Type = DistributionType_TrowbridgeReitz;
        microfacet.R = R;
        microfacet.fresnel_conductor.etaI = float3(1.0, 1.0, 1.0);
        microfacet.fresnel_conductor.etaT = eta;
        microfacet.fresnel_conductor.k = k;
        microfacet.trowbridgereitz_distribution.alpha_x = urough;
        microfacet.trowbridgereitz_distribution.alpha_y = vrough;
        microfacet.trowbridgereitz_distribution.sample_visible_area = true;
    }
    
    float3 f(float3 wo, float3 wi)
    {
        return microfacet.f(wo, wi);
    }
    
    float3 Samplef(float3 wo, Sampler _sampler, out float3 wi, out float pdf)
    {
        return microfacet.Samplef(wo, _sampler, wi, pdf);
    }
};

////////////// Mirror Material //////////////
struct MirrorMaterial
{
    SpecularReflection specular;
    
    void Init(Material mat)
    {
        specular.R = mat.base_color.rgb;
    }
    
    float3 f(float3 wo, float3 wi)
    {
        return specular.f(wo, wi);
    }
    
    float3 Samplef(float3 wo, Sampler _sampler, out float3 wi, out float pdf)
    {
        return specular.Samplef(wo, _sampler, wi, pdf);
    }
};

////////////// Substrate Material //////////////
struct SubstrateMaterial
{
    float3 Kd;
    float3 Rs;
    float anisotropic;
    float roughness;
    
    FresnelBlend fresnel_blend;
    
    void Init(Material mat)
    {
        Kd = mat.base_color.rgb;
        Rs = mat.data;
        anisotropic = mat.anisotropic;
        roughness = mat.roughness;
        
        float aspect = sqrt(1.0 - anisotropic * 0.9);
        float urough = max(0.001, roughness / aspect);
        float vrough = max(0.001, roughness * aspect);
        
        fresnel_blend.Distribution_Type = DistributionType_TrowbridgeReitz;
        fresnel_blend.Rd = Kd;
        fresnel_blend.Rs = Rs;
        fresnel_blend.trowbridgereitz_distribution.alpha_x = urough;
        fresnel_blend.trowbridgereitz_distribution.alpha_y = vrough;
        fresnel_blend.trowbridgereitz_distribution.sample_visible_area = true;
    }
    
    float3 f(float3 wo, float3 wi)
    {
        return fresnel_blend.f(wo, wi);
    }
    
    float3 Samplef(float3 wo, Sampler _sampler, out float3 wi, out float pdf)
    {
        return fresnel_blend.Samplef(wo, _sampler, wi, pdf);
    }
};

////////////// Glass Material //////////////
struct GlassMaterial
{
    float refraction;
    float3 R;
    float3 T;
    float anisotropic;
    float roughness;
    
    FresnelSpecular fresnel_specular;
    MicrofacetReflection microfacet_reflection;
    MicrofacetTransmission microfacet_transmission;
    
    void Init(Material mat)
    {
        refraction = mat.transmission;
        R = mat.base_color.rgb;
        T = mat.data;
        anisotropic = mat.anisotropic;
        roughness = mat.roughness;

        float aspect = sqrt(1.0 - anisotropic * 0.9);
        float urough = max(0.001, roughness / aspect);
        float vrough = max(0.001, roughness * aspect);
        
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
    }
    
    float3 f(float3 wo, float3 wi)
    {
        float3 distribution = float3(0.0, 0.0, 0.0);
        
        if(IsBlack(R) && IsBlack(T))
        {
            return float3(0.0, 0.0, 0.0);
        }
        
        bool isSpecular = (roughness == 0.0);
        
        if(isSpecular)
        {
            distribution += fresnel_specular.f(wo, wi);
        }
        else
        {
            if(!IsBlack(R))
            {
                distribution += microfacet_reflection.f(wo, wi);
            }
            if (!IsBlack(T))
            {
                distribution += microfacet_transmission.f(wo, wi);
            }
        }
        return distribution;
    }
    
    float3 Samplef(float3 wo, Sampler _sampler, out float3 wi, out float pdf)
    {
        if (IsBlack(R) && IsBlack(T))
        {
            return float3(0.0, 0.0, 0.0);
        }
        
        float2 u = _sampler.Get2D();
        
        bool isSpecular = (roughness == 0.0);
        
        if(isSpecular)
        {
            return fresnel_specular.Samplef(wo, _sampler, wi, pdf);
        }
        else
        {
            if (!IsBlack(R) && IsBlack(T))
            {
                return microfacet_reflection.Samplef(wo, _sampler, wi, pdf);
            }
            else if (IsBlack(R) && !IsBlack(T))
            {
                return microfacet_transmission.Samplef(wo, _sampler, wi, pdf);
            }
            else
            {
                FresnelDielectric fresnel = { 1.0, refraction };
                float3 F = fresnel.Evaluate(CosTheta(wo));
                
                if (u.x < F.x)
                {
                    float3 distribution = microfacet_reflection.Samplef(wo, _sampler, wi, pdf);
                    pdf *= F.x;
                    return distribution;
                }
                else
                {
                    float3 distribution = microfacet_transmission.Samplef(wo, _sampler, wi, pdf);
                    pdf *= 1.0 - F.x;
                    return distribution;
                }
            }
        }
        return float3(0.0, 0.0, 0.0);
    }
};

////////////// BSDF //////////////
struct BSDF
{
    Material mat;
    
    void Init(Material mat_)
    {
        mat = mat_;
    }
    
    float3 f(float3 wo, float3 wi)
    {
        if (mat.material_type == BxDF_Matte)
        {
            MatteMaterial bsdf;
            bsdf.Init(mat);
            return bsdf.f(wo, wi);
        }
        else if (mat.material_type == BxDF_Plastic)
        {
            PlasticMaterial bsdf;
            bsdf.Init(mat);
            return bsdf.f(wo, wi);
        }
        else if (mat.material_type == BxDF_Metal)
        {
            MetalMaterial bsdf;
            bsdf.Init(mat);
            return bsdf.f(wo, wi);
        }
        else if (mat.material_type == BxDF_Mirror)
        {
            MirrorMaterial bsdf;
            bsdf.Init(mat);
            return bsdf.f(wo, wi);
        }
        else if (mat.material_type == BxDF_Substrate)
        {
            SubstrateMaterial bsdf;
            bsdf.Init(mat);
            return bsdf.f(wo, wi);
        }
        else if (mat.material_type == BxDF_Glass)
        {
            GlassMaterial bsdf;
            bsdf.Init(mat);
            return bsdf.f(wo, wi);
        }
        
        return float3(0.0, 0.0, 0.0);
    }
    
    float3 Samplef(float3 wo, Sampler _sampler, out float3 wi, out float pdf)
    {
        if (mat.material_type == BxDF_Matte)
        {
            MatteMaterial bsdf;
            bsdf.Init(mat);
            return bsdf.Samplef(wo, _sampler, wi, pdf);
        }
        else if (mat.material_type == BxDF_Plastic)
        {
            MatteMaterial bsdf;
            bsdf.Init(mat);
            return bsdf.Samplef(wo, _sampler, wi, pdf);
        }
        else if (mat.material_type == BxDF_Metal)
        {
            MetalMaterial bsdf;
            bsdf.Init(mat);
            return bsdf.Samplef(wo, _sampler, wi, pdf);
        }
        else if (mat.material_type == BxDF_Mirror)
        {
            MirrorMaterial bsdf;
            bsdf.Init(mat);
            return bsdf.Samplef(wo, _sampler, wi, pdf);
        }
        else if (mat.material_type == BxDF_Substrate)
        {
            SubstrateMaterial bsdf;
            bsdf.Init(mat);
            return bsdf.Samplef(wo, _sampler, wi, pdf);
        }
        else if (mat.material_type == BxDF_Glass)
        {
            GlassMaterial bsdf;
            bsdf.Init(mat);
            return bsdf.Samplef(wo, _sampler, wi, pdf);
        }
        
        return float3(0.0, 0.0, 0.0);
    }
};

#endif