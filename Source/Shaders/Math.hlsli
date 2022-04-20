#ifndef __MATH_HLSL__
#define __MATH_HLSL__

#include "Constants.hlsli"

/*void CreateCoordinateSystem(in float3 N, out float3 Nt, out float3 Nb)
{
    Nt = normalize(((abs(N.z) > 0.99999f) ? float3(-N.x * N.y, 1.0f - N.y * N.y, -N.y * N.z) :
                                            float3(-N.x * N.z, -N.y * N.z, 1.0f - N.z * N.z)));
    Nb = cross(Nt, N);
}*/

void CoordinateSystem(in float3 N, out float3 Nt, out float3 Nb)
{
    float sign = (N.z >= 0.0f) * 2.0f - 1.0f; // copysign(1.0f, v1.z); // No HLSL support yet
    float a = -1.0f / (sign + N.z);
    float b = N.x * N.y * a;
    Nt = float3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    Nb = float3(b, sign + N.y * N.y * a, -N.y);
}

// Sampling Disk
// Polar Mapping
float2 UniformSampleDisk(float2 u)
{
    float r = sqrt(u.x);
    float theta = 2 * PI * u.y;
    return r * float2(cos(theta), sin(theta));
}

// Concentric Mapping
float2 SampleConcentricDisk(float2 u)
{
    float2 uOffset = 2.0 * u - 1.0;

    if (uOffset.x == 0 && uOffset.y == 0)
    {
        return float2(0.0, 0.0);
    }

    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y))
    {
        r = uOffset.x;
        theta = PIOver4 * (uOffset.y / uOffset.x);
    }
    else
    {
        r = uOffset.y;
        theta = PIOver2 - PIOver4 * (uOffset.x / uOffset.y);
    }

    return r * float2(cos(theta), sin(theta));
}

// Sampling Hemisphere
float3 UniformSampleHemisphere(float2 u)
{
    float z = u.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2 * PI * u.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

float3 UniformSampleSphere(float2 u)
{
    float z = 1.0 - 2.0 * u.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2 * PI * u.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

float UniformHemispherePdf()
{
    return Inv2PI;
}

float3 SampleCosineHemisphere(float2 u)
{
    float2 d = SampleConcentricDisk(u);
    float z = sqrt(max(0, 1 - d.x * d.x - d.y * d.y));
    return float3(d.x, d.y, z);
}

float CosineHemispherePdf(float cosTheta)
{
    return cosTheta * InvPI;
}

// Multiple Importance Sampling
float BalanceHeuristic(int nf, float fPdf, int ng, float gPdf)
{
    return (nf * fPdf) / (nf * fPdf + ng * gPdf);
}

float PowerHeuristic(int nf, float fPdf, int ng, float gPdf)
{
    float f = nf * fPdf;
    float g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

float SphericalPhi(float3 v)
{
    float p = atan2(v.y, v.x);
    return p < 0 ? (p + 2 * PI) : p;
}

float SphericalTheta(float3 v)
{
    return acos(clamp(v.z, -1.0, 1.0));
}

float ErfInv(float x)
{
    float w, p;
    x = clamp(x, -.99999, .99999);
    w = -log((1 - x) * (1 + x));
    if (w < 5)
    {
        w = w - 2.5;
        p = 2.81022636e-08;
        p = 3.43273939e-07 + p * w;
        p = -3.5233877e-06 + p * w;
        p = -4.39150654e-06 + p * w;
        p = 0.00021858087 + p * w;
        p = -0.00125372503 + p * w;
        p = -0.00417768164 + p * w;
        p = 0.246640727 + p * w;
        p = 1.50140941 + p * w;
    }
    else
    {
        w = sqrt(w) - 3;
        p = -0.000200214257;
        p = 0.000100950558 + p * w;
        p = 0.00134934322 + p * w;
        p = -0.00367342844 + p * w;
        p = 0.00573950773 + p * w;
        p = -0.0076224613 + p * w;
        p = 0.00943887047 + p * w;
        p = 1.00167406 + p * w;
        p = 2.83297682 + p * w;
    }
    return p * x;
}

float Erf(float x)
{
	// constants
    float a1 = 0.254829592f;
    float a2 = -0.284496736f;
    float a3 = 1.421413741f;
    float a4 = -1.453152027f;
    float a5 = 1.061405429f;
    float p = 0.3275911f;

	// Save the sign of x
    int sign = 1;
    if (x < 0)
    {
        sign = -1;
    }
    x = abs(x);

	// A&S formula 7.1.26
    float t = 1 / (1 + p * x);
    float y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return sign * y;
}

float AbsCosTheta(float3 w)
{
    return abs(w.z);
}

float CosTheta(float3 w)
{
    return w.z;
}

float Cos2Theta(float3 w)
{
    return w.z * w.z;
}

float Sin2Theta(float3 w)
{
    return max(0.0, 1.0 - Cos2Theta(w));
}

float Tan2Theta(float3 w)
{
    return Sin2Theta(w) / Cos2Theta(w);
}

float SinTheta(float3 w)
{
    return sqrt(Sin2Theta(w));
}

float SinPhi(float3 w)
{
    float sinTheta = SinTheta(w);
    return sinTheta == 0.0 ? 0.0 : clamp(w.y / sinTheta, -1.0, 1.0);
}

float Sin2Phi(float3 w)
{
    return SinPhi(w) * SinPhi(w);
}

float CosPhi(float3 w)
{
    float sinTheta = SinTheta(w);
    return sinTheta == 0.0 ? 1.0 : clamp(w.x / sinTheta, -1.0, 1.0);
}

float Cos2Phi(float3 w)
{
    return CosPhi(w) * CosPhi(w);
}

float TanTheta(float3 w)
{
    return SinTheta(w) / CosTheta(w);
}

float3 SphericalDirection(float sinTheta, float cosTheta, float phi)
{
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

float3 Faceforward(float3 v, float3 v2)
{
    return dot(v, v2) < 0.0 ? -v : v;
}

bool SameHemisphere(float3 w, float3 wp)
{
    return w.z * wp.z > 0;
}

bool Refract(float3 wi, float3 n, float eta, out float3 wt)
{
    float cosThetaI = dot(wi, n);
    float sin2ThetaI = max(0.0, 1.0 - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;

    if (sin2ThetaT >= 1.0)
    {
        return false;
    }

    float cosThetaT = sqrt(1.0 - sin2ThetaT);

    wt = eta * (-wi) + (eta * cosThetaI - cosThetaT) * n;

    return true;
}

float Radians(float deg)
{
    return PI / 180.0 * deg;
}

float Degrees(float rad)
{
    return rad * 180.0 / PI;
}

bool IsBlack(float3 v)
{
    return v.x == 0.0 && v.y == 0.0 && v.z == 0.0;
}



#endif