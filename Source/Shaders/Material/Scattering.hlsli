#ifndef MICROFACET_HLSLI
#define MICROFACET_HLSLI

#include "../Math.hlsli"
#include "../Interaction.hlsli"

//bool Refract(float3 wi, float3 n, float eta, out float3 wt)
//{
//    float cos_theta_i = dot(wi, n);
//    float sin_2_theta_i = max(0.0, 1.0 - cos_theta_i * cos_theta_i);
//    float sin_2_theta_t = eta * eta * sin_2_theta_i;
//
//    if (sin_2_theta_t >= 1.0)
//    {
//        return false;
//    }
//
//    float cos_theta_t = sqrt(1.0 - sin_2_theta_t);
//
//    wt = eta * (-wi) + (eta * cos_theta_i - cos_theta_t) * n;
//
//    return true;
//}

bool Refract(float3 wi, float3 n, float eta, out float etap, out float3 wt)
{
    float cos_theta_i = dot(n, wi);
    // Potentially flip interface orientation for Snell's law
    if (cos_theta_i < 0)
    {
        eta = 1 / eta;
        cos_theta_i = -cos_theta_i;
        n = -n;
    }

    // Compute $\cos\,\theta_\roman{t}$ using Snell's law
    float sin2Theta_i = max(0, 1 - Sqr(cos_theta_i));
    float sin2Theta_t = sin2Theta_i / Sqr(eta);
    // Handle total internal reflection case
    if (sin2Theta_t >= 1)
    {
        return false;
    }
    
    float cos_theta_t = sqrt(1 - sin2Theta_t);

    wt = -wi / eta + (cos_theta_i / eta - cos_theta_t) * n;
    // Provide relative IOR along ray to caller
    etap = eta;
    
    return true;
}

float3 Reflect(float3 wo, float3 n)
{
    return -wo + 2 * dot(wo, n) * n;
}

//float HenyeyGreenstein(float cos_theta, float g)
//{
//    float denom = 1 + Sqr(g) + 2 * g * cos_theta;
//    return Inv4PI * (1 - Sqr(g)) / (denom * sqrt(denom));
//}

//float FresnelDielectric(float cos_theta_i, float eta)
//{
//    cos_theta_i = clamp(cos_theta_i, -1, 1);
//    // Potentially flip interface orientation for Fresnel equations
//    if (cos_theta_i < 0)
//    {
//        eta = 1 / eta;
//        cos_theta_i = -cos_theta_i;
//    }
//
//    // Compute $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
//    float sin2Theta_i = 1 - Sqr(cos_theta_i);
//    float sin2Theta_t = sin2Theta_i / Sqr(eta);
//    if (sin2Theta_t >= 1)
//    {
//        return 1.f;
//    }
//    float cos_theta_t = sqrt(1 - sin2Theta_t);
//
//    float r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
//    float r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
//    return (Sqr(r_parl) + Sqr(r_perp)) / 2;
//}
//
//float FresnelComplex(float cos_theta_i, Complex eta)
//{
//    Complex cos_theta_i_ = ComplexFromReal(clamp(cos_theta_i, 0, 1));
//    // Compute complex $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
//    Complex sin2Theta_i = Sub(ComplexFromReal(1), Mul(cos_theta_i_, cos_theta_i_));
//    Complex sin2Theta_t = Div(sin2Theta_i, Mul(eta, eta));
//    Complex cos_theta_t = Sqrt(Sub(ComplexFromReal(1), sin2Theta_t));
//    
//    Complex r_parl = Div(Sub(Mul(eta, cos_theta_i_), cos_theta_t), Add(Mul(eta, cos_theta_i_), cos_theta_t));
//    Complex r_perp = Div(Sub(cos_theta_i_, Mul(eta, cos_theta_t)), Add(Mul(eta, cos_theta_t), cos_theta_i_));
//
//    return (Norm(r_parl) + Norm(r_perp)) / 2;
//}
//
//float3 FresnelComplex(float cos_theta_i, float3 eta, float3 k)
//{
//    float3 result;
//    result.x = FresnelComplex(cos_theta_i, CreateComplex(eta.x, k.x));
//    result.y = FresnelComplex(cos_theta_i, CreateComplex(eta.y, k.y));
//    result.z = FresnelComplex(cos_theta_i, CreateComplex(eta.z, k.z));
//    return result;
//}

//struct TrowbridgeReitzDistribution
//{
//    float alpha_x, alpha_y;
//    
//    bool EffectivelySmooth()
//    {
//        return max(alpha_x, alpha_y) < 1e-3f;
//    }
//    
//    void Init(float ax, float ay)
//    {
//        alpha_x = ax;
//        alpha_y = ay;
//        if (!EffectivelySmooth())
//        {
//            alpha_x = max(alpha_x, 1e-4f);
//            alpha_y = max(alpha_y, 1e-4f);
//        }
//    }
//    
//    float D(float3 wm)
//    {
//        float tan_2_theta = Tan2Theta(wm);
//        if (isinf(tan_2_theta))
//        {
//            return 0;
//        }
//        float cos_4_theta = Sqr(Cos2Theta(wm));
//        if (cos_4_theta < 1e-16f)
//        {
//            return 0;
//        }
//        float e = tan_2_theta * (Sqr(CosPhi(wm) / alpha_x) + Sqr(SinPhi(wm) / alpha_y));
//        return 1 / (PI * alpha_x * alpha_y * cos_4_theta * Sqr(1 + e));
//    }
//    
//    float Lambda(float3 w)
//    {
//        float tan_2_theta = Tan2Theta(w);
//        if (isinf(tan_2_theta))
//        {
//            return 0;
//        }
//        float alpha2 = Sqr(CosPhi(w) * alpha_x) + Sqr(SinPhi(w) * alpha_y);
//        return (sqrt(1 + alpha2 * tan_2_theta) - 1) / 2;
//    }
//    
//    float G1(float3 w)
//    {
//        return 1 / (1 + Lambda(w));
//    }
//    
//    float G(float3 wo, float3 wi)
//    {
//        return 1 / (1 + Lambda(wo) + Lambda(wi));
//    }
//    
//    float D(float3 w, float3 wm)
//    {
//        return G1(w) / AbsCosTheta(w) * D(wm) * abs(dot(w, wm));
//    }
//    
//    float PDF(float3 w, float3 wm)
//    {
//        return D(w, wm);
//    }
//    
//    float3 Sample_wm(float3 w, float2 u)
//    {
//        // Reference: https://jcgt.org/published/0007/04/01/paper.pdf
//        float3 vh = normalize(float3(alpha_x * w.x, alpha_y * w.y, w.z));
//        
//        float len = vh.x * vh.x + vh.y * vh.y;
//        float3 T1 = len > 0 ? float3(-vh.y, -vh.x, 0.f) / sqrt(len) : float3(1, 0, 0);
//        float3 T2 = cross(vh, T1);
//        
//        float2 p = UniformSampleDisk(u);
//        float s = 0.5 * (1.0 + vh.z);
//        p.y = (1.0 - s) * sqrt(1.0 - p.x * p.x) + s * p.y;
//        float3 nh = p.x * T1 + p.y * T2 + sqrt(max(0, 1.0 - p.x * p.x - p.y * p.y)) * vh;
//        
//        return normalize(float3(alpha_x * nh.x, alpha_y * nh.y, max(0, nh.z)));
//    }
//    
//    float RoughnessToAlpha(float roughness)
//    {
//        return sqrt(roughness);
//    }
//    
//    void Regularize()
//    {
//        if (alpha_x < 0.3f)
//        {
//            alpha_x = clamp(2 * alpha_x, 0.1f, 0.3f);
//        }
//        if (alpha_y < 0.3f)
//        {
//            alpha_y = clamp(2 * alpha_y, 0.1f, 0.3f);
//        }
//    }
//};

// Beckmann Distribution
void BeckmannSample11(float cos_theta_i, float u1, float u2, out float slope_x, out float slope_y)
{
    if (cos_theta_i > .9999)
    {
        float r = sqrt(-log(1.0 - u1));
        float sin_phi = sin(2 * PI * u2);
        float cos_phi = cos(2 * PI * u2);
        slope_x = r * cos_phi;
        slope_y = r * sin_phi;
        return;
    }

    float sin_theta_i = sqrt(max(0.0, 1.0 - cos_theta_i * cos_theta_i));
    float tan_theta_i = sin_theta_i / cos_theta_i;
    float cot_theta_i = 1.0 / tan_theta_i;

    float a = -1.0;
    float c = Erf(cot_theta_i);
    float sample_x = max(u1, float(1e-6));

    float theta_i = acos(cos_theta_i);
    float fit = 1 + theta_i * (-0.876 + theta_i * (0.4265 - 0.0594 * theta_i));
    float b = c - (1 + c) * pow(1 - sample_x, fit);

    const float sqrt_inv_PI = 1.0 / sqrt(PI);
    float normalization = 1.0 / (1.0 + c + sqrt_inv_PI * tan_theta_i * exp(-cot_theta_i * cot_theta_i));

    int it = 0;
    while (++it < 10)
    {
        if (!(b >= a && b <= c))
        {
            b = 0.5 * (a + c);
        }

        float inv_erf = ErfInv(b);
        float value = normalization * (1 + b + sqrt_inv_PI * tan_theta_i * exp(-inv_erf * inv_erf)) - sample_x;
        float derivative = normalization * (1 - inv_erf * tan_theta_i);

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
    slope_y = ErfInv(2.0 * max(u2, float(1e-6)) - 1.0);
}

float3 BeckmannSample(float3 wi, float alpha_x, float alpha_y, float u1, float u2)
{
	// stretch wi
    float3 wiStretched = normalize(float3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

	// simulate P22_{wi}(x_slope, y_slope, 1, 1)
    float slope_x, slope_y;
    BeckmannSample11(CosTheta(wiStretched), u1, u2, slope_x, slope_y);

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

struct BeckmannDistribution
{
    float alpha_x;
    float alpha_y;
    bool sample_visible_area;

    bool EffectivelySmooth() 
    { 
        return max(alpha_x, alpha_y) < 1e-3f; 
    }
    
    float D(float3 wm)
    {
        float tan_2_theta = Tan2Theta(wm);
        if (isinf(tan_2_theta))
        {
            return 0.0;
        }

        float cos_4_theta = Cos2Theta(wm) * Cos2Theta(wm);
        return exp(-tan_2_theta * (Cos2Phi(wm) / (alpha_x * alpha_x) + Sin2Phi(wm) / (alpha_y * alpha_y))) / (PI * alpha_x * alpha_y * cos_4_theta);
    }
    
    float Lambda(float3 w)
    {
        float abs_tan_theta = abs(TanTheta(w));
        if (isinf(abs_tan_theta))
        {
            return 0.;
        }

        float alpha = sqrt(Cos2Phi(w) * alpha_x * alpha_x + Sin2Phi(w) * alpha_y * alpha_y);
        float a = 1.0 / (alpha * abs_tan_theta);
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
    
    float Pdf(float3 wo, float3 wm)
    {
        if (sample_visible_area)
        {
            return D(wm) * G1(wo) * abs(dot(wo, wm)) / AbsCosTheta(wo);
        }
        else
        {
            return D(wm) * AbsCosTheta(wm);
        }
    }
    
    float3 SampleWm(float3 wo, float2 u)
    {
        if (!sample_visible_area)
        {
            float tan_2_theta, phi;
            if (alpha_x == alpha_y)
            {
                float log_sample = log(1 - u.x);
                tan_2_theta = -alpha_x * alpha_x * log_sample;
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

                float sin_phi = sin(phi);
                float cos_phi = cos(phi);
                float alphax2 = alpha_x * alpha_x;
                float alphay2 = alpha_y * alpha_y;
                tan_2_theta = -log_sample / (cos_phi * cos_phi / alphax2 + sin_phi * sin_phi / alphay2);
            }

            float cos_theta = 1.0 / sqrt(1.0 + tan_2_theta);
            float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
            float3 wm = SphericalDirection(sin_theta, cos_theta, phi);
            if (!SameHemisphere(wo, wm))
            {
                wm = -wm;
            }
            return wm;
        }
        else
        {
            float3 wm;
            bool flip = wo.z < 0.0;
            wm = BeckmannSample(flip ? -wo : wo, alpha_x, alpha_y, u.x, u.y);
            if (flip)
            {
                wm = -wm;
            }

            return wm;
        }

        return float3(0.0, 0.0, 0.0);
    }
};

// Trowbridge Reitz Distribution
void TrowbridgeReitzSample11(float cos_theta, float u1, float u2, out float slope_x, out float slope_y)
{
    if (cos_theta > .9999)
    {
        float r = sqrt(u1 / (1 - u1));
        float phi = 6.28318530718 * u2;
        slope_x = r * cos(phi);
        slope_y = r * sin(phi);
        return;
    }

    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    float tan_theta = sin_theta / cos_theta;
    float a = 1.0 / tan_theta;
    float G1 = 2.0 / (1.0 + sqrt(1.0 + 1.0 / (a * a)));

    float A = 2 * u1 / G1 - 1;
    float tmp = 1.f / (A * A - 1.f);
    if (tmp > 1e10)
    {
        tmp = 1e10;
    }
    float B = tan_theta;
    float D = sqrt(max(float(B * B * tmp * tmp - (A * A - B * B) * tmp), float(0)));
    float slope_x_1 = B * tmp - D;
    float slope_x_2 = B * tmp + D;
    slope_x = (A < 0 || slope_x_2 > 1.f / tan_theta) ? slope_x_1 : slope_x_2;

    float S;
    if (u2 > 0.5)
    {
        S = 1.0;
        u2 = 2.0 * (u2 - 0.5);
    }
    else
    {
        S = -1.0;
        u2 = 2.0 * (0.5 - u2);
    }

    float z =
	    (u2 * (u2 * (u2 * 0.27385 - 0.73369) + 0.46341)) /
	    (u2 * (u2 * (u2 * 0.093073 + 0.309420) - 1.000000) + 0.597999);

    slope_y = S * z * sqrt(1.0 + slope_x * slope_x);
}

float3 TrowbridgeReitzSample(float3 wi, float alpha_x, float alpha_y, float u1, float u2)
{
	// stretch wi
    float3 wiStretched = normalize(float3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

	// simulate P22_{wi}(x_slope, y_slope, 1, 1)
    float slope_x, slope_y;
    TrowbridgeReitzSample11(CosTheta(wiStretched), u1, u2, slope_x, slope_y);

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

struct TrowbridgeReitzDistribution
{
    float alpha_x;
    float alpha_y;
    bool sample_visible_area;

    bool EffectivelySmooth() 
    { 
        return max(alpha_x, alpha_y) < 1e-3f; 
    }
    
    static float RoughnessToAlpha(float roughness)
    {
        if(roughness <= float(1e-3))
        {
            return roughness;
        }
        roughness = max(roughness, (float) 1e-3);
        float x = log(roughness);
        return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
           0.000640711f * x * x * x * x;
    }
    
    float D(float3 wm)
    {
        float tan_2_theta = Tan2Theta(wm);

        if (isinf(tan_2_theta))
        {
            return 0.0;
        }
        const float cos_4_theta = Cos2Theta(wm) * Cos2Theta(wm);

        float e = (Cos2Phi(wm) / (alpha_x * alpha_x) + Sin2Phi(wm) / (alpha_y * alpha_y)) * tan_2_theta;
        return 1 / (PI * alpha_x * alpha_y * cos_4_theta * (1 + e) * (1 + e));
    }
    
    float3 SampleWm(float3 wo, float2 u)
    {
        float3 wm = 0.f;

        if (!sample_visible_area)
        {
            float cos_theta = 0.0;
            float phi = 2.0 * PI * u.y;

            if (alpha_x == alpha_y)
            {
                float tan_theta_2 = alpha_x * alpha_x * u.x * (1.0 - u.x);
                cos_theta = 1.0 / sqrt(1.0 + tan_theta_2);
            }
            else
            {
                phi = atan(alpha_y / alpha_x * tan(2.0 * PI * u.y + 0.5 * PI));
                if (u.y > 0.5)
                {
                    phi += PI;
                }
                float sin_phi = sin(phi);
                float cos_phi = cos(phi);

                const float alpha_x2 = alpha_x * alpha_x;
                const float alpha_y2 = alpha_y * alpha_y;

                const float alpha_2 = 1.0 / (cos_phi * cos_phi / alpha_x2 + sin_phi * sin_phi / alpha_y2);
                float tan_theta_2 = alpha_2 * u.x / (1.0 - u.x);
                cos_theta = 1.0 / sqrt(1 + tan_theta_2);
            }

            float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
            float3 wm = SphericalDirection(sin_theta, cos_theta, phi);
            if (!SameHemisphere(wo, wm))
            {
                wm = -wm;
            }
        }
        else
        {
            bool flip = wo.z < 0.0;
            wm = TrowbridgeReitzSample(flip ? -wo : wo, alpha_x, alpha_y, u.x, u.y);
            if (flip)
            {
                wm = -wm;
            }
        }
        return wm;
    }

    float Lambda(float3 w)
    {
        float abs_tan_theta = abs(TanTheta(w));
        if (isinf(abs_tan_theta))
        {
            return 0.;
        }
        float alpha = sqrt(Cos2Phi(w) * alpha_x * alpha_x + Sin2Phi(w) * alpha_y * alpha_y);
        float alpha_2_tan_2_theta = (alpha * abs_tan_theta) * (alpha * abs_tan_theta);
        return (-1.0 + sqrt(1.0 + alpha_2_tan_2_theta)) / 2.0;
    }

    float G1(float3 w)
    {
        return 1.0 / (1.0 + Lambda(w));
    }

    float G(float3 wo, float3 wi)
    {
        return 1.0 / (1.0 + Lambda(wo) + Lambda(wi));
    }

    float Pdf(float3 wo, float3 wm)
    {
        if (sample_visible_area)
        {
            return D(wm) * G1(wo) * abs(dot(wo, wm)) / AbsCosTheta(wo);
        }
        else
        {
            return D(wm) * AbsCosTheta(wm);
        }
    }
};

struct DisneyMicrofacetDistribution
{
    float alpha_x;
    float alpha_y;
    bool sample_visible_area;

    bool EffectivelySmooth() 
    { 
        return max(alpha_x, alpha_y) < 1e-3f; 
    }
    
    float D(float3 wh)
    {
        float tan_2_theta = Tan2Theta(wh);

        if (isinf(tan_2_theta))
        {
            return 0.0;
        }
        const float cos_4_theta = Cos2Theta(wh) * Cos2Theta(wh);

        float e = (Cos2Phi(wh) / (alpha_x * alpha_x) + Sin2Phi(wh) / (alpha_y * alpha_y)) * tan_2_theta;
        return 1 / (PI * alpha_x * alpha_y * cos_4_theta * (1 + e) * (1 + e));
    }
    
    float3 SampleWh(float3 wo, float2 u)
    {
        float3 wh = float3(0.0, 0.0, 0.0);
        
        if (!sample_visible_area)
        {
            float cos_theta = 0.0;
            float phi = 2.0 * PI * u.y;

            if (alpha_x == alpha_y)
            {
                float tan_theta_2 = alpha_x * alpha_x * u.x * (1.0 - u.y);
                cos_theta = 1.0 / sqrt(1.0 + tan_theta_2);
            }
            else
            {
                phi = atan(alpha_y / alpha_x * tan(2.0 * PI * u.y + 0.5 * PI));
                if (u.y > 0.5)
                {
                    phi += PI;
                }
                float sin_phi = sin(phi);
                float cos_phi = cos(phi);

                const float alpha_x2 = alpha_x * alpha_x;
                const float alpha_y2 = alpha_y * alpha_y;

                const float alpha_2 = 1.0 / (cos_phi * cos_phi / alpha_x2 + sin_phi * sin_phi / alpha_y2);
                float tan_theta_2 = alpha_2 * u.x / (1.0 - u.x);
                cos_theta = 1.0 / sqrt(1 + tan_theta_2);
            }

            float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
            float3 wh = SphericalDirection(sin_theta, cos_theta, phi);
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
        float abs_tan_theta = abs(TanTheta(w));
        if (isinf(abs_tan_theta))
        {
            return 0.;
        }
        float alpha = sqrt(Cos2Phi(w) * alpha_x * alpha_x + Sin2Phi(w) * alpha_y * alpha_y);
        float alpha_2_tan_2_theta = (alpha * abs_tan_theta) * (alpha * abs_tan_theta);
        return (-1.0 + sqrt(1.0 + alpha_2_tan_2_theta)) / 2.0;
    }

    float G1(float3 w)
    {
        return 1.0 / (1.0 + Lambda(w));
    }

    float G(float3 wo, float3 wi)
    {
        return G1(wo) * G1(wi);
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

#endif