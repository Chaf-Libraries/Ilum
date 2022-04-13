#ifndef __BXDF_HLSL__
#define __BXDF_HLSL__

#include "Math.hlsli"
#include "Common.hlsli"
#include "Random.hlsli"

#define USE_PCG
#ifdef USE_PCG
typedef PCGSampler Sampler;
#endif

static const uint DistributionType_Beckmann = 1 << 0;
static const uint DistributionType_TrowbridgeReitz = 1 << 1;

static const uint FresnelType_Conductor = 1 << 0;
static const uint FresnelType_Dielectric = 1 << 1;
static const uint FresnelType_Op = 1 << 2;

static const uint TransportMode_Radiance = 1 << 0;
static const uint TransportMode_Importance = 1 << 1;

static const uint BSDF_REFLECTION = 1 << 0;
static const uint BSDF_TRANSMISSION = 1 << 1;
static const uint BSDF_DIFFUSE = 1 << 2;
static const uint BSDF_GLOSSY = 1 << 3;
static const uint BSDF_SPECULAR = 1 << 4;
static const uint BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION | BSDF_TRANSMISSION;

static const uint BxDF_OrenNayar = 1 << 0;
static const uint BxDF_LambertianReflection = 1 << 1;
static const uint BxDF_MicrofacetReflection = 1 << 2;
static const uint BxDF_SpecularReflection = 1 << 3;
static const uint BxDF_SpecularTransmission = 1 << 4;
static const uint BxDF_FresnelBlend = 1 << 5;
static const uint BxDF_FresnelSpecular = 1 << 6;
static const uint BxDF_MicrofacetTransmission = 1 << 7;
static const uint BxDF_DisneyDiffuse = 1 << 8;
static const uint BxDF_DisneyFakeSS = 1 << 9;
static const uint BxDF_DisneyRetro = 1 << 10;
static const uint BxDF_DisneySheen = 1 << 11;

////////////// Beckmann Sample //////////////
void BeckmannSample11(float cosThetaI, float U1, float U2, out float slope_x, out float slope_y)
{
    if (cosThetaI > .9999)
    {
        float r = sqrt(-log(1.0 - U1));
        float sinPhi = sin(2 * PI * U2);
        float cosPhi = cos(2 * PI * U2);
        slope_x = r * cosPhi;
        slope_y = r * sinPhi;
        return;
    }

    float sinThetaI = sqrt(max(0.0, 1.0 - cosThetaI * cosThetaI));
    float tanThetaI = sinThetaI / cosThetaI;
    float cotThetaI = 1.0 / tanThetaI;

    float a = -1.0;
    float c = Erf(cotThetaI);
    float sample_x = max(U1, float(1e-6));

    float thetaI = acos(cosThetaI);
    float fit = 1 + thetaI * (-0.876 + thetaI * (0.4265 - 0.0594 * thetaI));
    float b = c - (1 + c) * pow(1 - sample_x, fit);

    const float sqrt_inv_PI = 1.0 / sqrt(PI);
    float normalization = 1.0 / (1.0 + c + sqrt_inv_PI * tanThetaI * exp(-cotThetaI * cotThetaI));

    int it = 0;
    while (++it < 10)
    {
        if (!(b >= a && b <= c))
        {
            b = 0.5 * (a + c);
        }

        float invErf = ErfInv(b);
        float value = normalization * (1 + b + sqrt_inv_PI * tanThetaI * exp(-invErf * invErf)) - sample_x;
        float derivative = normalization * (1 - invErf * tanThetaI);

        if (abs(value) < 1e-5)
        {
            break;
        }

        if (value > 0)
        {
            c = b;
        }
        else
        {
            a = b;
        }

        b -= value / derivative;
    }

    slope_x = ErfInv(b);
    slope_y = ErfInv(2.0 * max(U2, float(1e-6)) - 1.0);
}

float3 BeckmannSample(float3 wi, float alpha_x, float alpha_y, float U1, float U2)
{
	// stretch wi
    float3 wiStretched = normalize(float3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

	// simulate P22_{wi}(x_slope, y_slope, 1, 1)
    float slope_x, slope_y;
    BeckmannSample11(CosTheta(wiStretched), U1, U2, slope_x, slope_y);

	// rotate
    float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
    slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
    slope_x = tmp;

	// unstretch
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;

	// compute normal
    return normalize(float3(-slope_x, -slope_y, 1.0));
}

////////////// Trowbridge Reitz Sample //////////////
void TrowbridgeReitzSample11(float cosTheta, float U1, float U2, out float slope_x, out float slope_y)
{
    if (cosTheta > .9999)
    {
        float r = sqrt(U1 / (1 - U1));
        float phi = 6.28318530718 * U2;
        slope_x = r * cos(phi);
        slope_y = r * sin(phi);
        return;
    }

    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float tanTheta = sinTheta / cosTheta;
    float a = 1.0 / tanTheta;
    float G1 = 2.0 / (1.0 + sqrt(1.0 + 1.0 / (a * a)));

    float A = 2 * U1 / G1 - 1;
    float tmp = 1.f / (A * A - 1.f);
    if (tmp > 1e10)
    {
        tmp = 1e10;
    }
    float B = tanTheta;
    float D = sqrt(
	    max(float(B * B * tmp * tmp - (A * A - B * B) * tmp), float(0)));
    float slope_x_1 = B * tmp - D;
    float slope_x_2 = B * tmp + D;
    slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

    float S;
    if (U2 > 0.5)
    {
        S = 1.0;
        U2 = 2.0 * (U2 - 0.5);
    }
    else
    {
        S = -1.0;
        U2 = 2.0 * (0.5 - U2);
    }

    float z =
	    (U2 * (U2 * (U2 * 0.27385 - 0.73369) + 0.46341)) /
	    (U2 * (U2 * (U2 * 0.093073 + 0.309420) - 1.000000) + 0.597999);

    slope_y = S * z * sqrt(1.0 + slope_x * slope_x);
}

float3 TrowbridgeReitzSample(float3 wi, float alpha_x, float alpha_y, float U1, float U2)
{
	// stretch wi
    float3 wiStretched = normalize(float3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

	// simulate P22_{wi}(x_slope, y_slope, 1, 1)
    float slope_x, slope_y;
    TrowbridgeReitzSample11(CosTheta(wiStretched), U1, U2, slope_x, slope_y);

	// rotate
    float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
    slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
    slope_x = tmp;

	// unstretch
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;

	// compute normal
    return normalize(float3(-slope_x, -slope_y, 1.0));
}

////////////// Microfacet Roughness to Alpha //////////////
float RoughnessToAlpha(float roughness)
{
    roughness = max(roughness, float(1e-3));
    float x = log(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x +
	       0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}

////////////// Schlick Fresnel approximation //////////////
float SchlickWeight(float cosTheta)
{
    float m = clamp(1 - cosTheta, 0.0, 1.0);
    return pow(m, 5.0);
}

float FrSchlick(float R0, float cosTheta)
{
    return lerp(R0, 1.0, SchlickWeight(cosTheta));
}

float3 FrSchlick(float3 R0, float cosTheta)
{
    return lerp(R0, float3(1.0, 1.0, 1.0), SchlickWeight(cosTheta));
}

float SchlickR0FromEta(float eta)
{
    return ((eta - 1.0) * (eta - 1.0)) / ((eta + 1.0) * (eta + 1.0));
}

////////////// Fresnel //////////////
struct FresnelConductor
{
    float3 etaI;
    float3 etaT;
    float3 k;
        
    float3 Evaluate(float cosThetaI)
    {
        cosThetaI = clamp(cosThetaI, -1.0, 1.0);

        float3 eta = etaT / etaI;
        float3 etak = k / etaI;

        float cosThetaI2 = cosThetaI * cosThetaI;
        float sinThetaI2 = 1.0 - cosThetaI2;
        float3 eta2 = eta * eta;
        float3 etak2 = etak * etak;

        float3 t0 = eta2 - etak2 - sinThetaI2;
        float3 a2plusb2 = sqrt(t0 * t0 + 4.0 * eta2 * etak2);
        float3 t1 = a2plusb2 + cosThetaI2;
        float3 a = sqrt(0.5 * (a2plusb2 + t0));
        float3 t2 = 2.0 * cosThetaI * a;
        float3 Rs = (t1 - t2) / (t1 + t2);

        float3 t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
        float3 t4 = t2 * sinThetaI2;
        float3 Rp = Rs * (t3 - t4) / (t3 + t4);

        return 0.5 * (Rp + Rs);
    }
};

struct FresnelDielectric
{
    float etaI;
    float etaT;
    
    float3 Evaluate(float cosThetaI)
    {
        cosThetaI = clamp(cosThetaI, -1.0, 1.0);

	// Potentially swap indices of refraction
        if (cosThetaI <= 0.f)
        {
		// Swap
            float temp = etaI;
            etaI = etaT;
            etaT = temp;
            cosThetaI = abs(cosThetaI);
        }

        float sinThetaI = sqrt(max(0.0, 1.0 - cosThetaI * cosThetaI));
        float sinThetaT = etaI / etaT * sinThetaI;
        if (sinThetaT >= 1.0)
        {
            return float3(1.0, 1.0, 1.0);
        }

        float cosThetaT = sqrt(max(0.0, 1.0 - sinThetaT * sinThetaT));
        float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
	              ((etaT * cosThetaI) + (etaI * cosThetaT));
        float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
	              ((etaI * cosThetaI) + (etaT * cosThetaT));

        float R = Rparl * Rparl + Rperp * Rperp;
        
        return float3(R, R, R) / 2;
    }
};

struct FresnelOp
{
    float3 Evaluate(float cosThetaI)
    {
        return float3(1.0, 1.0, 1.0);
    }
};

////////////// Lambertian Reflection //////////////
struct LambertianReflection
{
    float3 R;
    static const uint BxDF_Type = BSDF_REFLECTION | BSDF_DIFFUSE;
        
    float3 f(float3 wo, float3 wi)
    {
        return R * InvPI;
    }
    
    float Pdf(float3 wo, float3 wi)
    {
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPI : 0.0;
    }
    
    float3 Samplef(float3 wo, float2 u, out float3 wi, out float pdf)
    {
        wi = SampleCosineHemisphere(u);
        if (wo.z < 0.0)
        {
            wi.z *= -1.0;
        }
        pdf = Pdf(wo, wi);
        return f(wo, wi);
    }
};

////////////// OrenNayar Reflection //////////////
struct OrenNayar
{
    float3 R;
    float A, B;
    static const uint BxDF_Type = BSDF_REFLECTION | BSDF_DIFFUSE;

    void Init(float3 R_, float sigma)
    {
        R = R_;
        sigma = Radians(sigma);
        float sigma2 = sigma * sigma;
        A = 1.0 - sigma2 / (2.0 * (sigma2 + 0.33));
        B = 0.45 * sigma2 / (sigma2 + 0.09);
    }
    
    float3 f(float3 wo, float3 wi)
    {
        float sinThetaI = SinTheta(wi);
        float sinThetaO = SinTheta(wo);

        float maxCos = 0;
        if (sinThetaI > 1e-4 && sinThetaO > 1e-4)
        {
            float sinPhiI = SinPhi(wi);
            float cosPhiI = CosPhi(wi);
            float sinPhiO = SinPhi(wo);
            float cosPhiO = CosPhi(wo);

            float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
            maxCos = max(0.0, dCos);
        }

        float tanThetaI = sinThetaI / AbsCosTheta(wi);
        float tanThetaO = sinThetaO / AbsCosTheta(wo);

        float sinAlpha, tanBeta;
        if (AbsCosTheta(wi) > AbsCosTheta(wo))
        {
            sinAlpha = sinThetaO;
            tanBeta = sinThetaI / AbsCosTheta(wi);
        }
        else
        {
            sinAlpha = sinThetaI;
            tanBeta = sinThetaO / AbsCosTheta(wo);
        }
        return R * InvPI * (A + B * maxCos * sinAlpha * tanBeta);
    }
    
    float Pdf(float3 wo, float3 wi)
    {
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPI : 0.0;
    }
    
    float3 Samplef(float3 wo, float2 u, out float3 wi, out float pdf)
    {
        wi = SampleCosineHemisphere(u);
        if (wo.z < 0.0)
        {
            wi.z *= -1.0;
        }
        pdf = Pdf(wo, wi);
        return f(wo, wi);
    }
};

////////////// Beckmann Distribution //////////////
struct BeckmannDistribution
{
    float alpha_x;
    float alpha_y;
    bool sample_visible_area;
    
    float D(float3 wh)
    {
        float tan2Theta = Tan2Theta(wh);
        if (isinf(tan2Theta))
        {
            return 0.0;
        }

        float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
        return exp(-tan2Theta * (Cos2Phi(wh) / (alpha_x * alpha_x) +
	                         Sin2Phi(wh) / (alpha_y * alpha_y))) /
	       (PI * alpha_x * alpha_y * cos4Theta);
    }
    
    float3 SampleWh(float3 wo, float2 u)
    {
        if (!sample_visible_area)
        {
            float tan2Theta, phi;
            if (alpha_x == alpha_y)
            {
                float log_sample = log(1 - u.x);
                tan2Theta = -alpha_x * alpha_x * log_sample;
                phi = u.y * 2 * PI;
            }
            else
            {
                float log_sample = log(1 - u.x);
                phi = atan(alpha_y / alpha_x * tan(2.0 * PI * u.y + 0.5 * PI));
                if (u.y > 0.5)
                {
                    phi += PI;
                }

                float sinPhi = sin(phi);
                float cosPhi = cos(phi);
                float alphax2 = alpha_x * alpha_x;
                float alphay2 = alpha_y * alpha_y;
                tan2Theta = -log_sample / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
            }

            float cosTheta = 1.0 / sqrt(1.0 + tan2Theta);
            float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
            float3 wh = SphericalDirection(sinTheta, cosTheta, phi);
            if (!SameHemisphere(wo, wh))
            {
                wh = -wh;
            }
            return wh;
        }
        else
        {
            float3 wh;
            bool flip = wo.z < 0.0;
            wh = BeckmannSample(flip ? -wo : wo, alpha_x, alpha_y, u.x, u.y);
            if (flip)
            {
                wh = -wh;
            }

            return wh;
        }

        return float3(0.0, 0.0, 0.0);
    }
    
    float Lambda(float3 w)
    {
        float absTanTheta = abs(TanTheta(w));
        if (isinf(absTanTheta))
        {
            return 0.;
        }

        float alpha = sqrt(Cos2Phi(w) * alpha_x * alpha_x + Sin2Phi(w) * alpha_y * alpha_y);
        float a = 1.0 / (alpha * absTanTheta);
        if (a >= 1.6)
        {
            return 0;
        }
        return (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a);
    }
    
    float G1(float3 w)
    {
        return 1.0 / (1.0 + Lambda(w));
    }

    float G(float3 wo, float3 wi)
    {
        return 1.0 / (1.0 + Lambda(wo) + Lambda(wi));
    }

    float Pdf(float3 wo, float3 wh)
    {
        if (sample_visible_area)
        {
            return D(wh) * G1(wo) * abs(dot(wo, wh)) / AbsCosTheta(wo);
        }
        else
        {
            return D(wh) * AbsCosTheta(wh);
        }
    }
};

////////////// Trowbridge Reitz Distribution //////////////
struct TrowbridgeReitzDistribution
{
    float alpha_x;
    float alpha_y;
    bool sample_visible_area;
    
    float D(float3 wh)
    {
        float tan2Theta = Tan2Theta(wh);

        if (isinf(tan2Theta))
        {
            return 0.0;
        }
        const float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

        float e = (Cos2Phi(wh) / (alpha_x * alpha_x) + Sin2Phi(wh) / (alpha_y * alpha_y)) * tan2Theta;
        return 1 / (PI * alpha_x * alpha_y * cos4Theta * (1 + e) * (1 + e));
    }
    
    float3 SampleWh(float3 wo, float2 u)
    {
        float3 wh = float3(0.0, 0.0, 0.0);
        

        if (!sample_visible_area)
        {
            float cosTheta = 0.0;
            float phi = 2.0 * PI * u.y;

            if (alpha_x == alpha_y)
            {
                float tanTheta2 = alpha_x * alpha_x * u.x * (1.0 - u.y);
                cosTheta = 1.0 / sqrt(1.0 + tanTheta2);
            }
            else
            {
                phi = atan(alpha_y / alpha_x * tan(2.0 * PI * u.y + 0.5 * PI));
                if (u.y > 0.5)
                {
                    phi += PI;
                }
                float sinPhi = sin(phi);
                float cosPhi = cos(phi);

                const float alpha_x2 = alpha_x * alpha_x;
                const float alpha_y2 = alpha_y * alpha_y;

                const float alpha_2 = 1.0 / (cosPhi * cosPhi / alpha_x2 + sinPhi * sinPhi / alpha_y2);
                float tanTheta2 = alpha_2 * u.x / (1.0 - u.x);
                cosTheta = 1.0 / sqrt(1 + tanTheta2);
            }

            float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
            float3 wh = SphericalDirection(sinTheta, cosTheta, phi);
            if (!SameHemisphere(wo, wh))
            {
                wh = -wh;
            }
        }
        else
        {
            bool flip = wo.z < 0.0;
            wh = TrowbridgeReitzSample(flip ? -wo : wo, alpha_x, alpha_y, u.x, u.y);
            if (flip)
            {
                wh = -wh;
            }
        }
        return wh;
    }

    float Lambda(float3 w)
    {
        float absTanTheta = abs(TanTheta(w));
        if (isinf(absTanTheta))
        {
            return 0.;
        }
        float alpha = sqrt(Cos2Phi(w) * alpha_x * alpha_x + Sin2Phi(w) * alpha_y * alpha_y);
        float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
        return (-1.0 + sqrt(1.0 + alpha2Tan2Theta)) / 2.0;
    }

    float G1(float3 w)
    {
        return 1.0 / (1.0 + Lambda(w));
    }

    float G(float3 wo, float3 wi)
    {
        return 1.0 / (1.0 + Lambda(wo) + Lambda(wi));
    }

    float Pdf(float3 wo, float3 wh)
    {
        if (sample_visible_area)
        {
            return D(wh) * G1(wo) * abs(dot(wo, wh)) / AbsCosTheta(wo);
        }
        else
        {
            return D(wh) * AbsCosTheta(wh);
        }
    }
};

////////////// Microfacet Reflection //////////////
struct MicrofacetReflection
{
    uint Distribution_Type;
    uint Fresnel_Type;
    static const uint BxDF_Type = BSDF_REFLECTION | BSDF_GLOSSY;
    
    float3 R;
    
    FresnelConductor fresnel_conductor;
    FresnelDielectric fresnel_dielectric;
    FresnelOp fresnel_op;
    
    BeckmannDistribution beckmann_distribution;
    TrowbridgeReitzDistribution trowbridgereitz_distribution;
        
    float3 f(float3 wo, float3 wi)
    {
        float cosThetaO = AbsCosTheta(wo);
        float cosThetaI = AbsCosTheta(wi);

        float3 wh = wi + wo;

        if (cosThetaI == 0.0 || cosThetaO == 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }
        if (IsBlack(wh))
        {
            return float3(0.0, 0.0, 0.0);
        }

        wh = normalize(wh);
        
        float3 F = float3(0.0, 0.0, 0.0);
        float D = 0.0;
        float G = 0.0;
        
        if (Distribution_Type == DistributionType_Beckmann)
        {
            D = beckmann_distribution.D(wh);
            G = beckmann_distribution.G(wo, wi);
        }
        else if (Distribution_Type == DistributionType_TrowbridgeReitz)
        {
            D = trowbridgereitz_distribution.D(wh);
            G = trowbridgereitz_distribution.G(wo, wi);
        }

        if (Fresnel_Type == FresnelType_Conductor)
        {
            F = fresnel_conductor.Evaluate(dot(wi, Faceforward(wh, float3(0.0, 0.0, 1.0))));
        }
        else if (Fresnel_Type == FresnelType_Dielectric)
        {
            F = fresnel_dielectric.Evaluate(dot(wi, Faceforward(wh, float3(0.0, 0.0, 1.0))));
        }
        else if (Fresnel_Type == FresnelType_Op)
        {
            F = fresnel_op.Evaluate(dot(wi, Faceforward(wh, float3(0.0, 0.0, 1.0))));
        }
        return R * D * G * F / (4.0 * cosThetaI * cosThetaO);
    }
    
    float Pdf(float3 wo, float3 wi)
    {
        if (!SameHemisphere(wo, wi))
        {
            return 0.0;
        }
        
        float3 wh = normalize(wo + wi);

        if (Distribution_Type == DistributionType_Beckmann)
        {
            return beckmann_distribution.Pdf(wo, wh) / (4.0 * dot(wo, wh));
        }
        else if (Distribution_Type == DistributionType_TrowbridgeReitz)
        {
            return trowbridgereitz_distribution.Pdf(wo, wh) / (4.0 * dot(wo, wh));
        }
        
        return 0.0;
    }
    
    float3 Samplef(float3 wo, float2 u, out float3 wi, out float pdf)
    {
        if (wo.z < 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }
        
        float3 wh = float3(0.0, 0.0, 0.0);
        
        if (Distribution_Type == DistributionType_Beckmann)
        {
            wh = beckmann_distribution.SampleWh(wo, u);
        }
        else if (Distribution_Type == DistributionType_TrowbridgeReitz)
        {
            wh = trowbridgereitz_distribution.SampleWh(wo, u);
        }

        if (dot(wo, wh) < 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }

        wi = reflect(-wo, wh);

        if (!SameHemisphere(wo, wi))
        {
            return float3(0.0, 0.0, 0.0);
        }
                
        if (Distribution_Type == DistributionType_Beckmann)
        {
            pdf = beckmann_distribution.Pdf(wo, wh) / (4.0 * dot(wo, wh));
        }
        else if (Distribution_Type == DistributionType_TrowbridgeReitz)
        {
            pdf = trowbridgereitz_distribution.Pdf(wo, wh) / (4.0 * dot(wo, wh));
        }
        
        return f(wo, wi);
    }
};

////////////// Specular Reflection //////////////
struct SpecularReflection
{
    float3 R;
    static const uint BxDF_Type = BSDF_REFLECTION | BSDF_SPECULAR;
    
    float3 f(float3 wo, float3 wi)
    {
        return float3(0.0, 0.0, 0.0);
    }
    
    float Pdf(float3 wo, float3 wi)
    {
        return 0.0;
    }
    
    float3 Samplef(float3 wo, float2 u, out float3 wi, out float pdf)
    {
        wi = float3(-wo.x, -wo.y, wo.z);
        pdf = 1.0;
                
        return R / AbsCosTheta(wi);
    }
};

////////////// Fresnel Blend //////////////
struct FresnelBlend
{
    float3 Rd;
    float3 Rs;
    
    static const uint BxDF_Type = BSDF_REFLECTION | BSDF_GLOSSY;
    
    uint Distribution_Type;
        
    BeckmannDistribution beckmann_distribution;
    TrowbridgeReitzDistribution trowbridgereitz_distribution;
    
    float3 SchlickFresnel(float cosTheta)
    {
        return Rs + pow(1 - cosTheta, 5.0) * (float3(1.0, 1.0, 1.0) - Rs);
    }
    
    float3 f(float3 wo, float3 wi)
    {
        float3 diffuse = (28.0 / (23.0 * PI)) * Rd * (1.0 - Rs) *
	               (1.0 - pow(1.0 - 0.5 * AbsCosTheta(wi), 5.0)) *
	               (1.0 - pow(1.0 - 0.5 * AbsCosTheta(wo), 5.0));

        float3 wh = wi + wo;

        if (wh.x == 0.0 && wh.y == 0.0 && wh.z == 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }
        wh = normalize(wh);
        
        float D = 0.0;
        
        if (Distribution_Type == DistributionType_Beckmann)
        {
            D = beckmann_distribution.D(wh);
        }
        else if (Distribution_Type == DistributionType_TrowbridgeReitz)
        {
            D = trowbridgereitz_distribution.D(wh);
        }

        float3 specular = D / (4.0 * abs(dot(wi, wh) * max(AbsCosTheta(wi), AbsCosTheta(wo)))) *
	                SchlickFresnel(dot(wi, wh));

        return diffuse + specular;
    }
    
    float Pdf(float3 wo, float3 wi)
    {
        if (!SameHemisphere(wo, wi))
        {
            return 0.0;
        }

        float3 wh = normalize(wo + wi);
        
        float pdf_wh = 0.0;
        
        if (Distribution_Type == DistributionType_Beckmann)
        {
            pdf_wh = beckmann_distribution.Pdf(wo, wh);
        }
        else if (Distribution_Type == DistributionType_TrowbridgeReitz)
        {
            pdf_wh = trowbridgereitz_distribution.Pdf(wo, wh);
        }

        return 0.5 * (AbsCosTheta(wi) * InvPI + pdf_wh / (4.0 * dot(wo, wh)));
    }
    
    float3 Samplef(float3 wo, float2 u, out float3 wi, out float pdf)
    {
        if (u.x < 0.5)
        {
            u.x = min(2.0 * u.x, 0.999999);
            wi = SampleCosineHemisphere(u);
            if (wo.z < 0.0)
            {
                wi.z *= -1.0;
            }
        }
        else
        {
            u.x = min(2.0 * (u.x - 0.5), 0.999999);
            float3 wh = float3(0.0, 0.0, 0.0);
        
            if (Distribution_Type == DistributionType_Beckmann)
            {
                wh = beckmann_distribution.SampleWh(wo, u);
            }
            else if (Distribution_Type == DistributionType_TrowbridgeReitz)
            {
                wh = trowbridgereitz_distribution.SampleWh(wo, u);
            }
            
            wi = reflect(wo, wh);
            if (!SameHemisphere(wo, wi))
            {
                return float3(0.0, 0.0, 0.0);
            }
        }

        pdf = Pdf(wo, wi);
        return f(wo, wi);
    }
};

////////////// Specular Transmission //////////////
struct SpecularTransmission
{
    float3 T;
    float etaA, etaB;
    FresnelDielectric fresnel;
    uint mode;
    
    static const uint BxDF_Type = BSDF_TRANSMISSION | BSDF_SPECULAR;
    
    float3 f(float3 wo, float3 wi)
    {
        return float3(0.0, 0.0, 0.0);
    }
    
    float Pdf(float3 wo, float3 wi)
    {
        return 0.0;
    }
    
    float3 Samplef(float3 wo, float2 u, out float3 wi, out float pdf)
    {
        bool entering = CosTheta(wo) > 0.0;
        float etaI = entering ? etaA : etaB;
        float etaT = entering ? etaB : etaA;

        if (!Refract(wo, Faceforward(float3(0.0, 0.0, 1.0), wo), etaI / etaT, wi))
        {
            return float3(0.0, 0.0, 0.0);
        }

        pdf = 1.0;

        float3 ft = T * (float3(0.0, 0.0, 0.0) - fresnel.Evaluate(CosTheta(wi)));

        if (mode == TransportMode_Radiance)
        {
            ft *= (etaI * etaI) / (etaT * etaT);
        }

        return ft / AbsCosTheta(wi);
    }
};

////////////// Microfacet Transmission //////////////
struct MicrofacetTransmission
{
    float3 T;
    float etaA, etaB;
    uint mode;
    
    static const uint BxDF_Type = BSDF_TRANSMISSION | BSDF_GLOSSY;
    
    uint Distribution_Type;
    uint Fresnel_Type;
    
    FresnelConductor fresnel_conductor;
    FresnelDielectric fresnel_dielectric;
    FresnelOp fresnel_op;
    
    BeckmannDistribution beckmann_distribution;
    TrowbridgeReitzDistribution trowbridgereitz_distribution;
    
    float3 f(float3 wo, float3 wi)
    {
        if (SameHemisphere(wo, wi))
        {
            return float3(0.0, 0.0, 0.0);
        }

        float cosThetaO = CosTheta(wo);
        float cosThetaI = CosTheta(wi);

        if (cosThetaO == 0.0 || cosThetaI == 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }

        float eta = CosTheta(wo) > 0.0 ? (etaB / etaA) : (etaA / etaB);
        float3 wh = normalize(wo + wi * eta);

        if (wh.z < 0.0)
        {
            wh = -wh;
        }
        if (dot(wo, wh) * dot(wi, wh) > 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }

        float3 F = float3(0.0, 0.0, 0.0);
        float D = 0.0;
        float G = 0.0;
        
        if (Distribution_Type == DistributionType_Beckmann)
        {
            D = beckmann_distribution.D(wh);
            G = beckmann_distribution.G(wo, wi);
        }
        else if (Distribution_Type == DistributionType_TrowbridgeReitz)
        {
            D = trowbridgereitz_distribution.D(wh);
            G = trowbridgereitz_distribution.G(wo, wi);
        }

        if (Fresnel_Type == FresnelType_Conductor)
        {
            F = fresnel_conductor.Evaluate(dot(wo, wh));
        }
        else if (Fresnel_Type == FresnelType_Dielectric)
        {
            F = fresnel_dielectric.Evaluate(dot(wo, wh));
        }
        else if (Fresnel_Type == FresnelType_Op)
        {
            F = fresnel_op.Evaluate(dot(wo, wh));
        }

        float sqrt_denom = dot(wo, wh) + eta * dot(wi, wh);
        float factor = (mode == TransportMode_Radiance) ? (1.0 / eta) : 1.0;

        return (float3(1.0, 1.0, 1.0) - F) * T *
	       abs(D * G * eta * eta * abs(dot(wi, wh)) * abs(dot(wo, wh)) * factor * factor /
	           (cosThetaI * cosThetaO * sqrt_denom * sqrt_denom));
    }
    
    float Pdf(float3 wo, float3 wi)
    {
        if (SameHemisphere(wo, wi))
        {
            return 0.0;
        }

        float eta = CosTheta(wo) > 0.0 ? (etaB / etaA) : (etaA / etaB);
        float3 wh = normalize(wo + wi * eta);

        if (dot(wo, wh) * dot(wi, wh) > 0.0)
        {
            return 0.0;
        }

        float sqrt_denom = dot(wo, wh) + eta * dot(wi, wh);
        float dwh_dwi = abs((eta * eta * dot(wi, wh)) / (sqrt_denom * sqrt_denom));

        float pdf = 0.0;
        
        if (Distribution_Type == DistributionType_Beckmann)
        {
            pdf = beckmann_distribution.Pdf(wo, wh);
        }
        else if (Distribution_Type == DistributionType_TrowbridgeReitz)
        {
            pdf = trowbridgereitz_distribution.Pdf(wo, wh);
        }

        return pdf * dwh_dwi;
    }
    
    float3 Samplef(float3 wo, float2 u, out float3 wi, out float pdf)
    {
        if (wo.z == 0.0)
        {
            return float3(0.0, 0.0, 0.0);
        }
        
        bool entering = CosTheta(wo) > 0.0;
        float etaI = entering ? etaA : etaB;
        float etaT = entering ? etaB : etaA;

        float3 wh = float3(0.0, 0.0, 0.0);
        
        if (Distribution_Type == DistributionType_Beckmann)
        {
            wh = beckmann_distribution.SampleWh(wo, u);
        }
        else if (Distribution_Type == DistributionType_TrowbridgeReitz)
        {
            wh = trowbridgereitz_distribution.SampleWh(wo, u);
        }

        if (!Refract(wo, Faceforward(wh, wo), etaI / etaT, wi))
        {
            return float3(0.0, 0.0, 0.0);
        }

        pdf = Pdf(wo, wi);

        return f(wo, wi);
    }
};

////////////// Fresnel Specular //////////////
struct FresnelSpecular
{
    float3 R;
    float3 T;
    float etaA;
    float etaB;
    uint mode;
    
    static const uint BxDF_Type = BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_SPECULAR;
    
    float3 f(float3 wo, float3 wi)
    {
        return float3(0.0, 0.0, 0.0);
    }
    
    float Pdf(float3 wo, float3 wi)
    {
        return 0.0;
    }
    
    float3 Samplef(float3 wo, float2 u, out float3 wi, out float pdf)
    {
        FresnelDielectric fresnel = { etaA, etaB };
        float3 F = fresnel.Evaluate(CosTheta(wo));

        if (u.x < F.x)
        {
            wi = float3(-wo.x, -wo.y, wo.z);
            pdf = F.x;
            return F * R / AbsCosTheta(wi);
        }
        else
        {
            bool entering = CosTheta(wo) > 0.0;
            float etaI = entering ? etaA : etaB;
            float etaT = entering ? etaB : etaA;

            if (!Refract(wo, Faceforward(float3(0.0, 0.0, 1.0), wo), etaI / etaT, wi))
            {
                return float3(0.0, 0.0, 0.0);
            }

            pdf = 1.0 - F.x;

            float3 ft = T * (float3(1.0, 1.0, 1.0) - F);

            if (mode == TransportMode_Radiance)
            {
                ft *= (etaI * etaI) / (etaT * etaT);
            }

            return ft / AbsCosTheta(wi);
        }
    }
};

////////////// Disney Diffuse //////////////
struct DisneyDiffuse
{
    float3 R;
    
    static const uint BxDF_Type = BSDF_REFLECTION | BSDF_DIFFUSE;
    
    float3 f(float3 wo, float3 wi)
    {
        float Fo = SchlickWeight(AbsCosTheta(wo));
        float Fi = SchlickWeight(AbsCosTheta(wi));
        return R * InvPI * (1.0 - Fo * 0.5) * (1.0 - Fi * 0.5);
    }
    
    float Pdf(float3 wo, float3 wi)
    {
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPI : 0;
    }
    
    float3 Samplef(float3 wo, float2 u, out float3 wi, out float pdf)
    {
        wi = SampleCosineHemisphere(u);
        if (wo.z < 0)
        {
            wi.z *= -1;
        }
        pdf = Pdf(wo, wi);
        return f(wo, wi);
    }
};

////////////// Disney FakeSS //////////////
struct DisneyFakeSS
{
    float3 R;
    float roughness;
    
    static const uint BxDF_Type = BSDF_REFLECTION | BSDF_DIFFUSE;
    
    float3 f(float3 wo, float3 wi)
    {
        float3 wh = wi + wo;
        if (IsBlack(wh))
        {
            return float3(0.0, 0.0, 0.0);
        }
        wh = normalize(wh);
        float cosThetaD = dot(wi, wh);
        float Fss90 = cosThetaD * cosThetaD * roughness;
        float Fo = SchlickWeight(AbsCosTheta(wo));
        float Fi = SchlickWeight(AbsCosTheta(wi));
        float Fss = lerp(1.0, Fss90, Fo) * lerp(1.0, Fss90, Fi);
        float ss = 1.25 * (Fss * (1.0 / (AbsCosTheta(wo) + AbsCosTheta(wi)) - 0.5) + 0.5);
        
        return R * InvPI * ss;
    }
    
    float Pdf(float3 wo, float3 wi)
    {
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPI : 0;
    }
    
    float3 Samplef(float3 wo, float2 u, out float3 wi, out float pdf)
    {
        wi = SampleCosineHemisphere(u);
        if (wo.z < 0)
        {
            wi.z *= -1;
        }
        pdf = Pdf(wo, wi);
        return f(wo, wi);
    }
};

////////////// Disney Retro //////////////
struct DisneyRetro
{
    float3 R;
    float roughness;
    
    static const uint BxDF_Type = BSDF_REFLECTION | BSDF_DIFFUSE;
    
    float3 f(float3 wo, float3 wi)
    {
        float3 wh = wi + wo;
        if (IsBlack(wh))
        {
            return float3(0.0, 0.0, 0.0);
        }
        wh = normalize(wh);
        float cosThetaD = dot(wi, wh);
        float Fss90 = cosThetaD * cosThetaD * roughness;
        float Fo = SchlickWeight(AbsCosTheta(wo));
        float Fi = SchlickWeight(AbsCosTheta(wi));
        float Rr = 2.0 * roughness * cosThetaD * cosThetaD;
        
        return R * InvPI * Rr * (Fo + Fi + Fo * Fi * (Rr - 1.0));
    }
    
    float Pdf(float3 wo, float3 wi)
    {
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPI : 0;
    }
    
    float3 Samplef(float3 wo, float2 u, out float3 wi, out float pdf)
    {
        wi = SampleCosineHemisphere(u);
        if (wo.z < 0)
        {
            wi.z *= -1;
        }
        pdf = Pdf(wo, wi);
        return f(wo, wi);
    }
};

////////////// Disney Sheen //////////////
struct DisneySheen
{
    float3 R;
    
    static const uint BxDF_Type = BSDF_REFLECTION | BSDF_DIFFUSE;
    
    float3 f(float3 wo, float3 wi)
    {
        float3 wh = wi + wo;
        if (IsBlack(wh))
        {
            return float3(0.0, 0.0, 0.0);
        }
        wh = normalize(wh);
        float cosThetaD = dot(wi, wh);
        return R * SchlickWeight(cosThetaD);
    }
    
    float Pdf(float3 wo, float3 wi)
    {
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPI : 0;
    }
    
    float3 Samplef(float3 wo, float2 u, out float3 wi, out float pdf)
    {
        wi = SampleCosineHemisphere(u);
        if (wo.z < 0)
        {
            wi.z *= -1;
        }
        pdf = Pdf(wo, wi);
        return f(wo, wi);
    }
};

#endif