#include "../../RayTracing.hlsli"

#define USE_DisneyDiffuse
#define USE_DisneyFakeSS
#define USE_SpecularTransmission
#define USE_DisneyRetro
#define USE_DisneySheen
#define USE_DisneyClearcoat
#define USE_MicrofacetReflection
#define USE_MicrofacetTransmission
#define USE_LambertianTransmission

#include "../../Material.hlsli"

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
    float lum = dot(float3(0.212671, 0.715160, 0.072169), c);
    float3 Ctint = lum > 0.0 ? (c / lum) : float3(1.0, 1.0, 1.0);
    float sheenWeight = isect.material.sheen;
    float3 Csheen = float3(0.0, 0.0, 0.0);
    if (sheenWeight > 0.0)
    {
        float stint = isect.material.sheen_tint;
        Csheen = lerp(float3(1.0, 1.0, 1.0), Ctint, stint);
    }
    
    if (diffuseWeight > 0.0)
    {
        if (isect.material.thin > 0)
        {
            float flat = isect.material.flatness;
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
    
    float aspect = sqrt(1.0 - isect.material.anisotropic * 0.9);
    float ax = max(0.001, sqrt(rough) / aspect);
    float ay = max(0.001, sqrt(rough) * aspect);
    
    float specTint = isect.material.specular_tint;
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
    float cc = isect.material.clearcoat;
    if (cc > 0.0)
    {
        // Add Disney Clear Coat
        bsdfs.AddBxDF(BxDF_DisneyClearcoat);
        disney_clearcoat.weight = cc;
        disney_clearcoat.gloss = lerp(0.1, 0.001, isect.material.clearcoat_gloss);
    }
    
    // BTDF
    if (strans > 0.0)
    {
        float3 T = strans * sqrt(c);
        if (isect.material.thin > 0.0)
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
    if (isect.material.thin > 0.0)
    {
        // Add Lambertian Transmission
        bsdfs.AddBxDF(BxDF_LambertianTransmission);
        lambertian_transmission.T = dt * c;
    }
    
    return bsdfs;
}

[shader("callable")]
void main(inout BSDFSampleDesc bsdf)
{
    BSDFs mat = CreateDisneyMaterial(bsdf.isect);
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
