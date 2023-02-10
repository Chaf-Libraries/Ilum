#ifndef PRINCIPLED_BSDF_HLSLI
#define PRINCIPLED_BSDF_HLSLI

#include "BSDF.hlsli"
#include "../Scattering.hlsli"
#include "../Fresnel.hlsli"
#include "../../Math.hlsli"
#include "../../Common.hlsli"
#include "../../Interaction.hlsli"

float2 CalculateAnisotropicRoughness(float anisotropic, float roughness)
{
    float rough2 = Sqr(roughness);
    if(anisotropic == 0)
    {
        float a = max(0.001, rough2);
        return a;
    }
    float aspect = sqrt(1.0 - 0.9 * anisotropic);
    return float2(max(0.001, rough2 / aspect), max(0.001, rough2 * aspect));
}

// Schlick Fresnel Approximation
float SchlickWeight(float cos_theta)
{
    float m = clamp(1 - cos_theta, 0.0, 1.0);
    return pow(m, 5.0);
}

float FrSchlick(float R0, float cos_theta)
{
    return lerp(R0, 1.0, SchlickWeight(cos_theta));
}

float3 FrSchlick(float3 R0, float cos_theta)
{
    return lerp(R0, float3(1.0, 1.0, 1.0), SchlickWeight(cos_theta));
}

float SchlickR0FromEta(float eta)
{
    return ((eta - 1.0) * (eta - 1.0)) / ((eta + 1.0) * (eta + 1.0));
}

// Diffuse
float3 DisneyDiffuse(float3 base_color, float3 wo, float3 wi)
{
    float Fo = SchlickWeight(AbsCosTheta(wo));
    float Fi = SchlickWeight(AbsCosTheta(wi));
    return base_color * InvPI * (1 - Fo / 2) * (1 - Fi / 2);
}

// Fake subsurface
float3 DisneyFakeSS(float3 base_color, float roughness, float3 wo, float3 wi)
{
    float3 wh = wi + wo;
    if(IsBlack(wh))
    {
        return 0.f;
    }
    wh = normalize(wh);
    float cos_theta_d = dot(wi, wh);
    float Fss90 = cos_theta_d * cos_theta_d * roughness;
    float Fo = SchlickWeight(AbsCosTheta(wo));
    float Fi = SchlickWeight(AbsCosTheta(wi));
    float Fss = lerp(1.0, Fss90, Fo) * lerp(1.0, Fss90, Fi);
    float ss = 1.25f * (Fss * (1 / (AbsCosTheta(wo) + AbsCosTheta(wi)) - 0.5) + 0.5f);

    return base_color * InvPI * ss;
}

// Retro
float3 DisneyRetro(float3 base_color, float roughness, float3 wo, float3 wi)
{
    float3 wh = wi + wo;
    if(IsBlack(wh))
    {
        return 0.f;
    }
    wh = normalize(wh);
    float cos_theta_d = dot(wi, wh);
    float Fo = SchlickWeight(AbsCosTheta(wo));
    float Fi = SchlickWeight(AbsCosTheta(wi));
    float Rr = 2 * roughness * cos_theta_d * cos_theta_d;

    return base_color * InvPI * Rr * (Fo + Fi + Fo * Fi * (Rr - 1));
}

// Sheen
float3 DisneySheen(float3 base_color, float lum, float sheen_tint, float3 wo, float3 wi)
{
    float3 wh = wi + wo;
    if(IsBlack(wh))
    {
        return 0.f;
    }
    wh = normalize(wh);
    float cos_theta_d = dot(wi, wh);

    float3 c_tint = lum > 0 ? base_color / lum : 1;
    float3 c_sheen = (1 - sheen_tint) + sheen_tint * c_tint;

    return c_sheen * SchlickWeight(cos_theta_d);
}

float GTR1(float cos_theta, float alpha)
{
    float alpha2 = alpha * alpha;
    return (alpha2 - 1) / (PI * log(alpha2) * (1 + (alpha2 - 1) * cos_theta * cos_theta));
}

float SmithG_GGX(float cos_theta, float alpha)
{
    float alpha2 = alpha * alpha;
    float cos_theta_2 = cos_theta * cos_theta;
    return 1 / (cos_theta + sqrt(alpha2 + cos_theta_2 - alpha2 * cos_theta_2));
}

// Clearcoat
float3 DisneyClearcoat(float gloss, float3 wo, float3 wi)
{
    float3 wh = wi + wo;
    if(IsBlack(wh))
    {
        return 0.f;
    }
    wh = normalize(wh);
    float cos_theta_i = AbsCosTheta(wi);
    float cos_theta_o = AbsCosTheta(wo);
    float Dcc = GTR1(AbsCosTheta(wh), lerp(0.1f, 0.001f, gloss));
    float Fcc = FrSchlick(0.04, dot(wo, wh));
    float Gcc = SmithG_GGX(AbsCosTheta(wo), 0.25) * SmithG_GGX(AbsCosTheta(wi), 0.25);

    return Fcc * Dcc * Gcc / (4 * cos_theta_i * cos_theta_o);
}

float DisneyClearcoatPdf(float gloss, float3 wo, float3 wi)
{
    float3 wh = wi + wo;
    if(IsBlack(wh))
    {
        return 0.f;
    }
    wh = normalize(wh);
    float Dr = GTR1(AbsCosTheta(wh), lerp(0.1f, 0.001f, gloss));
    return Dr * AbsCosTheta(wh) / (4 * dot(wo, wh));
}

float3 DisneyClearcoatSample(float gloss, float3 wo, float2 u)
{
    if(wo.z == 0)
    {
        return 0.f;
    }

    float alpha2 = gloss * gloss;
    float cos_theta = sqrt(max(0, (1 - pow(alpha2, 1 - u.x)) / (1 - alpha2)));
    float sin_theta = sqrt(max(0, 1 - cos_theta * cos_theta));
    float phi = 2 * PI * u.y;
    float3 wh = SphericalDirection(sin_theta, cos_theta, phi);
    if(!SameHemisphere(wo, wh))
    {
        wh = -wh;
    }

    float3 wi = Reflect(wo, wh);

    return wi;
}

struct PrincipledBSDF
{
    float3 base_color;
    float roughness;
    float anisotropic;
    float spec_trans;
    float eta;
    float sheen;
    float sheen_tint;
    float specular;
    float spec_tint;
    float metallic;
    float clearcoat;
    float clearcoat_gloss;
    float subsurface;

    void Init(
        float3 base_color_,
        float roughness_,
        float anisotropic_,
        float spec_trans_,
        float eta_,
        float sheen_,
        float sheen_tint_,
        float specular_,
        float spec_tint_,
        float metallic_,
        float clearcoat_,
        float clearcoat_gloss_,
        float subsurface_)
    {
        base_color = base_color_;
        roughness = roughness_;
        anisotropic = anisotropic_;
        spec_trans = spec_trans_;
        eta = eta_;
        sheen = sheen_;
        sheen_tint = sheen_tint_;
        specular = specular_;
        spec_tint = spec_tint_;
        metallic = metallic_;
        clearcoat = clearcoat_;
        clearcoat_gloss = clearcoat_gloss_;
        subsurface = subsurface_;
    }

    uint Flags()
    {
        return BSDF_DiffuseReflection;
    }

    float3 Eval(float3 wo, float3 wi, TransportMode mode)
    {
        float3 f = 0.f;

        float cos_theta_i = CosTheta(wi);
        float cos_theta_o = CosTheta(wo);

        float brdf = (1.f - metallic) * (1.f - spec_trans);
        float bsdf = (1.f - metallic) * spec_trans;

        bool has_reflect = (cos_theta_i * cos_theta_o > 0.f);
        bool has_refract = (cos_theta_i * cos_theta_o < 0.f);

        float etap = 1;
        if(has_refract)
        {
            etap = cos_theta_o > 0.f ? eta : (1 / eta);
        }

        float3 wm = wi * etap + wo;

        if(cos_theta_i == 0 || cos_theta_o == 0 || LengthSquared(wm) == 0)
        {
            return 0.f;
        }

        wm = FaceForward(normalize(wm), float3(0, 0, 1));

        bool front_side = cos_theta_o > 0.f;

        float F_dielectric = FresnelDielectric(CosTheta(wo), eta);

        bool has_sheen = (sheen > 0.f) && has_reflect && (1.f - metallic > 0.f) && front_side;
        bool has_diffuse = (brdf > 0.f) && has_reflect && front_side;
        bool has_spec_reflect = has_reflect && (F_dielectric > 0.f);
        bool has_clearcoat = has_reflect && (clearcoat > 0.f) && front_side;
        bool has_spec_trans = (bsdf > 0.f) && has_refract && (F_dielectric < 1.f);

        float roughness2 = Sqr(roughness);
        float2 alpha = max(0.001, roughness2);
        if(anisotropic > 0)
        {
            float aspect = sqrt(1.f - 0.9f * anisotropic);
            alpha.x = max(0.001, roughness2 / aspect);
            alpha.y = max(0.001, roughness2 * aspect);
        }
        TrowbridgeReitzDistribution spec_dist;
        spec_dist.alpha_x = alpha.x;
        spec_dist.alpha_y = alpha.y;
        spec_dist.sample_visible_area = true;

        float D = spec_dist.D(wm);
        float G = spec_dist.G1(wo) * spec_dist.G1(wi);
        float lum = Luminance(base_color);

        if(has_diffuse)
        {
            float3 f_diff = DisneyDiffuse(base_color, wo, wi);
            float3 f_retro = DisneyRetro(base_color, roughness, wo, wi);

            if(subsurface > 0)
            {
                float3 f_ss = DisneyFakeSS(base_color, roughness, wo, wi);
                f += brdf * lerp(f_diff + f_retro, f_ss, subsurface);
            }
            else
            {
                f += brdf * f_diff + f_retro;
            }
        }

        if(has_sheen)
        {
            f += sheen * (1.f - metallic) * DisneySheen(base_color, lum, sheen_tint, wo, wi);
        }

        if(has_clearcoat)
        {
            f += 0.25 * clearcoat * DisneyClearcoat(clearcoat_gloss, wo, wi);
        }

        if(has_spec_reflect)
        {
            float3 F_schlick = 0.f;
            if(metallic > 0)
            {
                F_schlick += metallic * FrSchlick(base_color, cos_theta_o);
            }
            if(spec_tint > 0)
            {
                float3 c_tint = lum > 0.f ? base_color / lum : 1.f;
                float3 F0_spec_tint = c_tint * SchlickR0FromEta(etap);
            }
            float3 F = front_side ? 
                (1.f - metallic) * (1.f - spec_tint) * F_dielectric + F_schlick :
                bsdf * F_dielectric;
            f += F * D * G / abs(4.f * cos_theta_i * cos_theta_o);
        }

        if(has_spec_trans)
        {
            float denom = Sqr(dot(wi, wm) + dot(wo, wm) / etap) * cos_theta_i * cos_theta_o;
            float3 ft = sqrt(base_color) * bsdf * D * (1.f - F_dielectric) * G * abs(dot(wi, wm) * dot(wo, wm) / denom);
            if(mode == TransportMode_Radiance)
            {
                ft /= Sqr(etap);
            }
            f += ft;
        }

        return f;
    }

    float PDF(float3 wo, float3 wi, TransportMode mode, SampleFlags flags)
    {
        float cos_theta_i = CosTheta(wi);
        float cos_theta_o = CosTheta(wo);

        float brdf = (1.f - metallic) * (1.f - spec_trans);
        float bsdf = (1.f - metallic) * spec_trans;

        bool has_reflect = (cos_theta_i * cos_theta_o > 0.f);
        bool has_refract = (cos_theta_i * cos_theta_o < 0.f);

        float etap = 1;
        if(has_refract)
        {
            etap = cos_theta_o > 0.f ? eta : (1 / eta);
        }

        float3 wm = wi * etap + wo;

        if(cos_theta_i == 0 || cos_theta_o == 0 || LengthSquared(wm) == 0)
        {
            return 0.f;
        }

        wm = FaceForward(normalize(wm), float3(0, 0, 1));

        bool front_side = cos_theta_o > 0.f;

        float F_dielectric = FresnelDielectric(CosTheta(wo), eta);

        bool has_sheen = (sheen > 0.f) && has_reflect && (1.f - metallic > 0.f) && front_side;
        bool has_diffuse = (brdf > 0.f) && has_reflect && front_side;
        bool has_spec_reflect = has_reflect && (F_dielectric > 0.f);
        bool has_clearcoat = has_reflect && (clearcoat > 0.f) && front_side;
        bool has_spec_trans = (bsdf > 0.f) && has_refract && (F_dielectric < 1.f);

        float prob_spec_reflect = front_side ? 1.f - bsdf * (1.f - F_dielectric) : F_dielectric;
        float prob_spec_trans = has_spec_trans ? (front_side ? bsdf * (1.f - F_dielectric) : 1.f - F_dielectric) : 0.f;
        float prob_clearcoat = has_clearcoat ? (front_side ? 0.25f * clearcoat : 0.f) : 0.f;
        float prob_diffuse = front_side ? brdf : 0.f;
        float rcp_tot_prob = rcp(prob_spec_reflect + prob_spec_trans + prob_clearcoat + prob_diffuse);
        
        prob_spec_reflect *= rcp_tot_prob;
        prob_spec_trans *= rcp_tot_prob;
        prob_clearcoat *= rcp_tot_prob;
        prob_diffuse *= rcp_tot_prob;

        float roughness2 = Sqr(roughness);
        float2 alpha = max(0.001, roughness2);
        if(anisotropic > 0)
        {
            float aspect = sqrt(1.f - 0.9f * anisotropic);
            alpha.x = max(0.001, roughness2 / aspect);
            alpha.y = max(0.001, roughness2 * aspect);
        }
        TrowbridgeReitzDistribution spec_dist;
        spec_dist.alpha_x = alpha.x;
        spec_dist.alpha_y = alpha.y;
        spec_dist.sample_visible_area = true;

        float pdf = 0.f;

        if(has_diffuse)
        {
            pdf += prob_diffuse * AbsCosTheta(wi) * InvPI;
        }

        if(has_clearcoat)
        {
            pdf += prob_clearcoat * DisneyClearcoatPdf(clearcoat_gloss, wo, wi);
        }

        if(has_spec_reflect)
        {
            pdf += prob_spec_reflect * spec_dist.Pdf(wo, wm) / (4.0 * abs(dot(wo, wm)));
        }

        if(has_spec_trans)
        {
            float denom = Sqr(dot(wi, wm) + dot(wo, wm) / etap);
            float dwm_dwi = abs(dot(wi, wm)) / denom;
            pdf += prob_spec_trans * spec_dist.Pdf(wo, wm) * dwm_dwi;
        }        
        
        return pdf;
    }

    BSDFSample Samplef(float3 wo, float uc, float2 u, TransportMode mode, SampleFlags flags)
    {
        BSDFSample bsdf_sample;
        bsdf_sample.Init();

        float cos_theta_o = CosTheta(wo);

        float brdf = (1.f - metallic) * (1.f - spec_trans);
        float bsdf = (1.f - metallic) * spec_trans;

        bool has_reflect = (flags & SampleFlags_Reflection);
        bool has_refract = (flags & SampleFlags_Transmission);

        bool front_side = cos_theta_o > 0.f;

        float F_dielectric = FresnelDielectric(CosTheta(wo), eta);

        bool has_sheen = (sheen > 0.f) && has_reflect && (1.f - metallic > 0.f) && front_side;
        bool has_diffuse = (brdf > 0.f) && has_reflect && front_side;
        bool has_spec_reflect = has_reflect && (F_dielectric > 0.f);
        bool has_clearcoat = has_reflect && (clearcoat > 0.f) && front_side;
        bool has_spec_trans = (bsdf > 0.f) && has_refract && (F_dielectric < 1.f);

        float prob_spec_reflect = front_side ? 1.f - bsdf * (1.f - F_dielectric) : F_dielectric;
        float prob_spec_trans = has_spec_trans ? (front_side ? bsdf * (1.f - F_dielectric) : 1.f - F_dielectric) : 0.f;
        float prob_clearcoat = has_clearcoat ? (front_side ? 0.25f * clearcoat : 0.f) : 0.f;
        float prob_diffuse = front_side ? brdf : 0.f;

        float rcp_tot_prob = rcp(prob_spec_reflect + prob_spec_trans + prob_clearcoat + prob_diffuse);
        prob_spec_reflect *= rcp_tot_prob;
        prob_spec_trans *= rcp_tot_prob;
        prob_clearcoat *= rcp_tot_prob;
        prob_diffuse *= rcp_tot_prob;

        if(uc < prob_spec_reflect)
        {
            float roughness2 = Sqr(roughness);
            float2 alpha = max(0.001, roughness2);
            if(anisotropic > 0)
            {
                float aspect = sqrt(1.f - 0.9f * anisotropic);
                alpha.x = max(0.001, roughness2 / aspect);
                alpha.y = max(0.001, roughness2 * aspect);
            }
            TrowbridgeReitzDistribution spec_dist;
            spec_dist.alpha_x = alpha.x;
            spec_dist.alpha_y = alpha.y;
            spec_dist.sample_visible_area = true;

            float3 wm = spec_dist.SampleWm(wo, u);

            // Sample specular reflection
            bsdf_sample.wi = Reflect(wo, FaceForward(wm, wo));
            bsdf_sample.flags = BSDF_GlossyReflection;
            bsdf_sample.eta = 1;
        }
        else if(uc < prob_spec_reflect + prob_spec_trans)
        {
            // Sample specular transmission
            float roughness2 = Sqr(roughness);
            float2 alpha = max(0.001, roughness2);
            if(anisotropic > 0)
            {
                float aspect = sqrt(1.f - 0.9f * anisotropic);
                alpha.x = max(0.001, roughness2 / aspect);
                alpha.y = max(0.001, roughness2 * aspect);
            }
            TrowbridgeReitzDistribution spec_dist;
            spec_dist.alpha_x = alpha.x;
            spec_dist.alpha_y = alpha.y;
            spec_dist.sample_visible_area = true;

            float3 wm = spec_dist.SampleWm(wo, u);

            float etap = 0.f;
            float3 wi;
            if (!Refract(wo, Faceforward(wm, float3(0, 0, 1)), eta, etap, wi))
            {
                return bsdf_sample;
            }
            bsdf_sample.wi = wi;
            bsdf_sample.flags = BSDF_GlossyReflection;
            bsdf_sample.eta = etap;
        }
        else if(uc < prob_spec_reflect + prob_spec_trans + prob_clearcoat)
        {
            // Sample clearcoat
            bsdf_sample.wi = DisneyClearcoatSample(clearcoat_gloss, wo, u);
            bsdf_sample.flags = BSDF_GlossyReflection;
            bsdf_sample.eta = 1;
        }
        else
        {
            // Sample diffuse
            bsdf_sample.wi = SampleCosineHemisphere(u);
            bsdf_sample.flags = BSDF_DiffuseReflection;
            bsdf_sample.eta = 1;
        }

        bsdf_sample.f = Eval(wo, bsdf_sample.wi, mode);
        bsdf_sample.pdf = PDF(wo, bsdf_sample.wi, mode, flags);

        return bsdf_sample;
    }

};

#endif