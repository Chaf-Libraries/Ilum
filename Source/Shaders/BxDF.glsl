#ifndef _BXDF_GLSL_
#define _BXDF_GLSL_

#include "Geometry.glsl"
#include "Random.glsl"
#include "Sampling.glsl"

/*
struct BxDFContext
{
	float NoV;
	float NoL;
	float VoL;
	float NoH;
	float VoH;
	float XoV;
	float XoL;
	float XoH;
	float YoV;
	float YoL;
	float YoH;
};

void InitBxDF(inout BxDFContext context, vec3 N, vec3 V, vec3 L)
{
	context.NoL   = dot(N, L);
	context.NoV   = dot(N, V);
	context.VoL   = dot(V, L);
	float InvLenH = sqrt(2 + 2 * context.VoL);
	context.NoH   = clamp((context.NoL + context.NoV) * InvLenH, 0.0, 1.0);
	context.VoH   = clamp(InvLenH + InvLenH * context.VoL, 0.0, 1.0);

	context.XoV = 0.0f;
	context.XoL = 0.0f;
	context.XoH = 0.0f;
	context.YoV = 0.0f;
	context.YoL = 0.0f;
	context.YoH = 0.0f;
}

void InitBxDF(inout BxDFContext context, vec3 N, vec3 X, vec3 Y, vec3 V, vec3 L)
{
	context.NoL   = dot(N, L);
	context.NoV   = dot(N, V);
	context.VoL   = dot(V, L);
	float InvLenH = sqrt(2 + 2 * context.VoL);
	context.NoH   = clamp((context.NoL + context.NoV) * InvLenH, 0.0, 1.0);
	context.VoH   = clamp(InvLenH + InvLenH * context.VoL, 0.0, 1.0);

	context.XoV = dot(X, V);
	context.XoL = dot(X, L);
	context.XoH = (context.XoL + context.XoV) * InvLenH;
	context.YoV = dot(Y, V);
	context.YoL = dot(Y, L);
	context.YoH = (context.YoL + context.YoV) * InvLenH;
}

// Physical based shading model
// Diffuse Term

// Lambert
vec3 Diffuse_Lambert(vec3 diffuse_color)
{
	return diffuse_color * (1 / PI);
}

// [Burley 2012, "Physically-Based Shading at Disney"]
vec3 Diffuse_Burley(vec3 diffuse_color, float roughness, float NoV, float NoL, float VoH)
{
	float FD90 = 0.5 + 2.0 * VoH * VoH * roughness;
	float FdV  = 1 + (FD90 - 1) * pow(1 - NoV, 5.0);
	float FdL  = 1 + (FD90 - 1) * pow(1 - NoL, 5.0);
	return diffuse_color * ((1 / PI) * FdV * FdL);
}

// [Gotanda 2012, "Beyond a Simple Physically Based Blinn-Phong Model in Real-Time"]
vec3 Diffuse_OrenNayar(vec3 diffuse_color, float roughness, float NoV, float NoL, float VoH)
{
	float a     = roughness * roughness;
	float s     = a;        // 1.29 + 0.5 * a
	float s2    = s * s;
	float VoL   = 2.0 * VoH * VoH - 1.0;
	float Cosri = VoL - NoV * NoL;
	float C1    = 1 - 0.5 * s2 / (s2 + 0.33);
	float C2    = 0.45 * s2 / (s2 + 0.09) * Cosri * (Cosri >= 0 ? 1 / (max(NoV, NoL)) : 1.0);
	return diffuse_color / PI * (C1 + C2) * (1 + roughness * 0.5);
}

// [Gotanda 2014, "Designing Reflectance Models for New Consoles"]
vec3 Diffuse_Gotanda(vec3 diffuse_color, float roughness, float NoV, float NoL, float VoH)
{
	float a     = roughness * roughness;
	float a2    = a * a;
	float F0    = 0.04;
	float VoL   = 2 * VoH * VoH - 1;        // double angle identity
	float Cosri = VoL - NoV * NoL;
	float a2_13 = a2 + 1.36053;
	float Fr    = (1 - (0.542026 * a2 + 0.303573 * a) / a2_13) * (1 - pow(1 - NoV, 5 - 4 * a2) / a2_13) * ((-0.733996 * a2 * a + 1.50912 * a2 - 1.16402 * a) * pow(1 - NoV, 1 + 1.0/(39 * a2 * a2 + 1)) + 1);
	//float Fr = ( 1 - 0.36 * a ) * ( 1 - pow( 1 - NoV, 5 - 4*a2 ) / a2_13 ) * ( -2.5 * Roughness * ( 1 - NoV ) + 1 );
	float Lm = (max(1 - 2 * a, 0) * (1 - pow(1 - NoL, 5.0)) + min(2 * a, 1)) * (1 - 0.5 * a * (NoL - 1)) * NoL;
	float Vd = (a2 / ((a2 + 0.09) * (1.31072 + 0.995584 * NoV))) * (1 - pow(1 - NoL, (1 - 0.3726732 * NoV * NoV) / (0.188566 + 0.38841 * NoV)));
	float Bp = Cosri < 0 ? 1.4 * NoV * NoL * Cosri : Cosri;
	float Lr = (21.0 / 20.0) * (1 - F0) * (Fr * Lm + Vd + Bp);
	return diffuse_color / PI * Lr;
}

// Normal Distribute Function

// [Blinn 1977, "Models of light reflection for computer synthesized pictures"]
float D_Blinn(float a2, float NoH)
{
	float n = 2 / a2 - 2;
	return (n + 2) / (2 * PI) * pow(NoH, n); 
}

// [Beckmann 1963, "The scattering of electromagnetic waves from rough surfaces"]
float D_Beckmann(float a2, float NoH)
{
	float NoH2 = NoH * NoH;
	return exp((NoH2 - 1) / (a2 * NoH2)) / (PI * a2 * NoH2 * NoH2);
}

// GGX / Trowbridge-Reitz
// [Walter et al. 2007, "Microfacet models for refraction through rough surfaces"]
float D_GGX(float a2, float NoH)
{
	float d = (NoH * a2 - NoH) * NoH + 1;        // 2 mad
	return a2 / (PI * d * d);                    // 4 mul, 1 rcp
}

// Anisotropic GGX
// [Burley 2012, "Physically-Based Shading at Disney"]
float D_GGXaniso(float ax, float ay, float NoH, float XoH, float YoH)
{
	float  a2 = ax * ay;
	vec3 V  = vec3(ay * XoH, ax * YoH, a2 * NoH);
	float  S  = dot(V, V);

	return (1.0f / PI) * a2 * pow(a2 / S, 2.0);
}

// Geometry Visibility Term
float Vis_Implicit()
{
	return 0.25;
}

// [Neumann et al. 1999, "Compact metallic reflectance models"]
float Vis_Neumann(float NoV, float NoL)
{
	return 1 / (4 * max(NoL, NoV));
}

// [Kelemen 2001, "A microfacet based coupled specular-matte brdf model with importance sampling"]
float Vis_Kelemen(float VoH)
{
	// constant to prevent NaN
	return 1.0 / (4 * VoH * VoH + 1e-5);
}

// Tuned to match behavior of Vis_Smith
// [Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"]
float Vis_Schlick(float a2, float NoV, float NoL)
{
	float k            = sqrt(a2) * 0.5;
	float Vis_SchlickV = NoV * (1 - k) + k;
	float Vis_SchlickL = NoL * (1 - k) + k;
	return 0.25 / (Vis_SchlickV * Vis_SchlickL);
}

// Smith term for GGX
// [Smith 1967, "Geometrical shadowing of a random rough surface"]
float Vis_Smith(float a2, float NoV, float NoL)
{
	float Vis_SmithV = NoV + sqrt(NoV * (NoV - NoV * a2) + a2);
	float Vis_SmithL = NoL + sqrt(NoL * (NoL - NoL * a2) + a2);
	return 1.0 / (Vis_SmithV * Vis_SmithL);
}

// Appoximation of joint Smith term for GGX
// [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
float Vis_SmithJointApprox(float a2, float NoV, float NoL)
{
	float a          = sqrt(a2);
	float Vis_SmithV = NoL * (NoV * (1 - a) + a);
	float Vis_SmithL = NoV * (NoL * (1 - a) + a);
	return 0.5 * 1.0 / (Vis_SmithV + Vis_SmithL);
}

// [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
float Vis_SmithJoint(float a2, float NoV, float NoL)
{
	float Vis_SmithV = NoL * sqrt(NoV * (NoV - NoV * a2) + a2);
	float Vis_SmithL = NoV * sqrt(NoL * (NoL - NoL * a2) + a2);
	return 0.5 * 1.0 / (Vis_SmithV + Vis_SmithL);
}

// [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
float Vis_SmithJointAniso(float ax, float ay, float NoV, float NoL, float XoV, float XoL, float YoV, float YoL)
{
	float Vis_SmithV = NoL * length(vec3(ax * XoV, ay * YoV, NoV));
	float Vis_SmithL = NoV * length(vec3(ax * XoL, ay * YoL, NoL));
	return 0.5 * 1.0 / (Vis_SmithV + Vis_SmithL);
}

vec3 F_None(vec3 specular_color)
{
	return specular_color;
}

// [Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"]
vec3 F_Schlick(vec3 specular_color, float VoH)
{
	float Fc = pow(1 - VoH, 5.0);        // 1 sub, 3 mul
	//return Fc + (1 - Fc) * specular_color;		// 1 add, 3 mad

	// Anything less than 2% is physically impossible and is instead considered to be shadowing
	return clamp(50.0 * specular_color.g, 0.0, 1.0) * Fc + (1 - Fc) * specular_color;
}

vec3 F_Fresnel(vec3 specular_color, float VoH)
{
	vec3 SpecularColorSqrt = sqrt(clamp(vec3(0, 0, 0), vec3(0.99, 0.99, 0.99), specular_color));
	vec3 n                 = (1 + SpecularColorSqrt) / (1 - SpecularColorSqrt);
	vec3 g                 = sqrt(n * n + VoH * VoH - 1);
	return 0.5 * pow((g - VoH) / (g + VoH), 2.0) * (1 + pow(((g + VoH) * VoH - 1) / ((g - VoH) * VoH + 1), 2.0));
}
*/

// Diffuse Term
// Lambertian Diffuse
vec3 LambertianDiffuse(vec3 albedo)
{
	return albedo / PI;
}

// Disney Diffuse
vec3 Diffuse_Burley_Disney(vec3 albedo, float roughness, float NoV, float NoL, float VoH)
{
	float FD90 = 0.5 + 2.0 * VoH * VoH * roughness;
	float FdV  = 1 + (FD90 - 1) * pow(1 - NoV, 5.0);
	float FdL  = 1 + (FD90 - 1) * pow(1 - NoL, 5.0);
	return albedo * ((1 / PI) * FdV * FdL);
}

// Normal Distribute Term
// Blinn Phong NDF
float DistributeBlinnPhong(float NoH, float roughness)
{
	float a2 = roughness * roughness;
	return pow(NoH, 2.0 / a2 - 2.0) / (PI * a2);
}

// Beckmann NDF
float DistributeBeckmann(float NoH, float roughness)
{
	float a2   = roughness * roughness;
	float NoH2 = NoH * NoH;
	return exp((NoH2 - 1.0) / (a2 * NoH2)) / (PI * a2 * NoH2 * NoH2 + 0.00001);
}

// GTR1 NDF aka Berry
float DistributeBerry(float NoH, float roughness)
{
	float alpha2 = roughness * roughness;
	float den    = 1.0 + (alpha2 - 1.0) * NoH * NoH;
	return (alpha2 - 1.0) / (PI * log(alpha2) * den);
}

// GTR2 NDF aka Trowbridge-Reitz(GGX)
float DistributeGGX(float NoH, float roughness)
{
	float alpha  = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom  = NoH * NoH * (alpha2 - 1.0) + 1.0;
	return alpha2 / (PI * denom * denom);
}

// Anisotropic GGX
float DistributeAnisotropicGGX(float HoX, float HoY, float NoH, float ax, float ay)
{
	float deno = HoX * HoX / (ax * ax) + HoY * HoY / (ay * ay) + NoH * NoH;
	return 1.0 / (PI * ax * ay * deno * deno);
}

// Geometry Term
// Smith GGX
float GeometrySmithGGX(float NoV, float alpha)
{
	float a = alpha * alpha;
	float b = NoV * NoV;
	return 1.0 / (NoV + sqrt(a + b - a * b));
}

// Anisotropic Smith GGX
float GeometryAnisotropicSmithGGX(float VoN, float VoX, float VoY, float ax, float ay)
{
	return 1.0 / (VoN + sqrt(pow(VoX * ax, 2.0) + pow(VoY * ay, 2.0) + pow(VoN, 2.0)));
}

// Schlick GGX
float GeometrySchlickGGX(float NoV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;
	return NoV / (NoV * (1.0 - k) + k);
}

float GeometrySmith(float NoL, float NoV, float roughness)
{
	float ggx1 = GeometrySchlickGGX(NoL, roughness);
	float ggx2 = GeometrySchlickGGX(NoV, roughness);

	return ggx1 * ggx2;
}

// Fresnel Term
// Schlick Fresnel
vec3 FresnelSchlick(float LoH, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(clamp(1 - LoH, 0, 1), 5.0);
}

vec3 AverageFresnel(vec3 r, vec3 g)
{
	return vec3(0.087237) + 0.0230685 * g - 0.0864902 * g * g + 0.0774594 * g * g * g + 0.782654 * r - 0.136432 * r * r + 0.278708 * r * r * r + 0.19744 * g * r + 0.0360605 * g * g * r - 0.2586 * g * r * r;
}

// Kulla Conty Multi-Scatter Approximation
vec3 MultiScatterBRDF(vec3 albedo, vec3 Eo, vec3 Ei, float Eavg)
{
	// copper
	vec3 edgetint = vec3(0.827, 0.792, 0.678);
	vec3 F_avg    = AverageFresnel(albedo, edgetint);

	vec3 f_add = F_avg * Eavg / (1.0 - F_avg * (1.0 - Eavg));
	vec3 f_ms  = (1.0 - Eo) * (1.0 - Ei) / (PI * (1.0 - Eavg.r));

	return f_ms * f_add;
}

// Disney BRDF
vec3 DisneyBRDF(vec3 L, vec3 V, vec3 N, vec3 X, vec3 Y, MaterialData material)
{
	float NoL = dot(N, L);
	float NoV = dot(N, V);

	if (NoL < 0.0 || NoV < 0.0)
	{
		return vec3(0.0);
	}

	vec3  H   = normalize(L + V);
	float NoH = dot(N, H);
	float LoH = dot(L, H);

	vec3 Cdlin = pow(material.base_color.rgb, vec3(2.2));
	// Luminace approximation
	float Cdlum = 0.3 * Cdlin.x + 0.1 * Cdlin.y + 0.1 * Cdlin.z;

	vec3 Ctint  = Cdlum > 0 ? Cdlin / Cdlum : vec3(1.0);
	vec3 Cspec0 = mix(material.specular * 0.08 * mix(vec3(1.0), Ctint, material.specular_tint), Cdlin, material.metallic);
	vec3 Csheen = mix(vec3(1.0), Ctint, material.sheen_tint);

	// Diffuse term
	float FL   = pow(clamp(1.0 - NoL, 0.0, 1.0), 5.0);
	float FV   = pow(clamp(1.0 - NoV, 0.0, 1.0), 5.0);
	float Fd90 = 0.5 + 2 * LoH * LoH * material.roughness;
	float Fd   = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);

	// Subsurface term
	// Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
	float Fss90 = LoH * LoH * material.roughness;
	float Fss   = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);
	float ss    = 1.25 * (Fss * (1.0 / (NoL + NoV) - 0.5) + 0.5);

	// Specular term
	float aspect = sqrt(1.0 - material.anisotropic * 0.9);
	float ax     = max(0.001, sqrt(material.roughness) / aspect);
	float ay     = max(0.001, sqrt(material.roughness) * aspect);
	//float HoX, float HoY, float NoH, float ax, float ay
	float Ds = DistributeAnisotropicGGX(dot(H, X), dot(H, Y), NoH, ax, ay);

	// Fresnel term
	vec3 Fs = FresnelSchlick(LoH, Cspec0);

	// Geometry term
	float Gs = GeometryAnisotropicSmithGGX(NoL, dot(L, X), dot(L, Y), ax, ay);
	Gs *= GeometryAnisotropicSmithGGX(NoV, dot(V, X), dot(V, Y), ax, ay);

	// Sheen
	float FH     = pow(clamp(1 - LoH, 0, 1), 5.0);
	vec3  Fsheen = FH * material.sheen * Csheen;

	// Clearcoat: ior = 1.5 -> F0 = 0.04
	float Dr = DistributeBerry(NoH, mix(0.1, 0.001, material.clearcoat_gloss));
	float Fr = mix(0.04, 1.0, FH);
	float Gr = GeometrySmithGGX(NoL, 0.25) * GeometrySmithGGX(NoV, 0.25);

	return ((1 / PI) * mix(Fd, ss, material.subsurface) * Cdlin + Fsheen) * (1 - material.metallic) + Gs * Fs * Ds + 0.25 * material.clearcoat * Gr * Fr * Dr;
}

// Cook Torrance BRDF
vec3 CookTorranceBRDF(vec3 L, vec3 V, vec3 N, vec3 X, vec3 Y, MaterialData material)
{
	vec3 Cdlin = pow(material.base_color.rgb, vec3(2.2));
	vec3 F0    = vec3(0.04);
	F0         = mix(F0, Cdlin, material.metallic);

	vec3  H   = normalize(V + L);
	float NoV = clamp(dot(N, V), 0.0, 1.0);
	float NoL = clamp(dot(N, L), 0.0, 1.0);
	float NoH = clamp(dot(N, H), 0.0, 1.0);
	float HoV = clamp(dot(H, V), 0.0, 1.0);

	float D = DistributeGGX(NoH, material.roughness);
	float G = GeometrySmith(NoL, NoV, material.roughness);
	vec3  F = FresnelSchlick(HoV, F0);

	vec3 specular = D * F * G / (4.0 * NoL * NoV + 0.001);
	vec3 Kd       = (vec3(1.0) - F) * (1 - material.metallic);

	return Kd * LambertianDiffuse(Cdlin) + specular;
}

////////////// Beckmann Sample //////////////
void BeckmannSample11(float cosThetaI, float U1, float U2, out float slope_x, out float slope_y)
{
	if (cosThetaI > .9999)
	{
		float r      = sqrt(-log(1.0 - U1));
		float sinPhi = sin(2 * PI * U2);
		float cosPhi = cos(2 * PI * U2);
		slope_x      = r * cosPhi;
		slope_y      = r * sinPhi;
		return;
	}

	float sinThetaI = sqrt(max(0.0, 1.0 - cosThetaI * cosThetaI));
	float tanThetaI = sinThetaI / cosThetaI;
	float cotThetaI = 1.0 / tanThetaI;

	float a        = -1.0;
	float c        = Erf(cotThetaI);
	float sample_x = max(U1, float(1e-6));

	float thetaI = acos(cosThetaI);
	float fit    = 1 + thetaI * (-0.876 + thetaI * (0.4265 - 0.0594 * thetaI));
	float b      = c - (1 + c) * pow(1 - sample_x, fit);

	const float sqrt_inv_PI   = 1.0 / sqrt(PI);
	float       normalization = 1.0 / (1.0 + c + sqrt_inv_PI * tanThetaI * exp(-cotThetaI * cotThetaI));

	int it = 0;
	while (++it < 10)
	{
		if (!(b >= a && b <= c))
		{
			b = 0.5 * (a + c);
		}

		float invErf     = ErfInv(b);
		float value      = normalization * (1 + b + sqrt_inv_PI * tanThetaI * exp(-invErf * invErf)) - sample_x;
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

vec3 BeckmannSample(vec3 wi, float alpha_x, float alpha_y, float U1, float U2)
{
	// stretch wi
	vec3 wiStretched = normalize(vec3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

	// simulate P22_{wi}(x_slope, y_slope, 1, 1)
	float slope_x, slope_y;
	BeckmannSample11(CosTheta(wiStretched), U1, U2, slope_x, slope_y);

	// rotate
	float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
	slope_y   = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
	slope_x   = tmp;

	// unstretch
	slope_x = alpha_x * slope_x;
	slope_y = alpha_y * slope_y;

	// compute normal
	return normalize(vec3(-slope_x, -slope_y, 1.0));
}

////////////// Trowbridge Reitz Sample //////////////
void TrowbridgeReitzSample11(float cosTheta, float U1, float U2, out float slope_x, out float slope_y)
{
	if (cosTheta > .9999)
	{
		float r   = sqrt(U1 / (1 - U1));
		float phi = 6.28318530718 * U2;
		slope_x   = r * cos(phi);
		slope_y   = r * sin(phi);
		return;
	}

	float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
	float tanTheta = sinTheta / cosTheta;
	float a        = 1.0 / tanTheta;
	float G1       = 2.0 / (1.0 + sqrt(1.0 + 1.0 / (a * a)));

	float A   = 2 * U1 / G1 - 1;
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
	slope_x         = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

	float S;
	if (U2 > 0.5)
	{
		S  = 1.0;
		U2 = 2.0 * (U2 - 0.5);
	}
	else
	{
		S  = -1.0;
		U2 = 2.0 * (0.5 - U2);
	}

	float z =
	    (U2 * (U2 * (U2 * 0.27385 - 0.73369) + 0.46341)) /
	    (U2 * (U2 * (U2 * 0.093073 + 0.309420) - 1.000000) + 0.597999);

	slope_y = S * z * sqrt(1.0 + slope_x * slope_x);
}

vec3 TrowbridgeReitzSample(vec3 wi, float alpha_x, float alpha_y, float U1, float U2)
{
	// stretch wi
	vec3 wiStretched = normalize(vec3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

	// simulate P22_{wi}(x_slope, y_slope, 1, 1)
	float slope_x, slope_y;
	TrowbridgeReitzSample11(CosTheta(wiStretched), U1, U2, slope_x, slope_y);

	// rotate
	float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
	slope_y   = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
	slope_x   = tmp;

	// unstretch
	slope_x = alpha_x * slope_x;
	slope_y = alpha_y * slope_y;

	// compute normal
	return normalize(vec3(-slope_x, -slope_y, 1.0));
}

////////////// Microfacet Roughness to Alpha //////////////
float RoughnessToAlpha(float roughness)
{
	roughness = max(roughness, float(1e-3));
	float x   = log(roughness);
	return 1.62142f + 0.819955f * x + 0.1734f * x * x +
	       0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}

////////////// Fresnel //////////////
struct FresnelConductor
{
	vec3 etaI;
	vec3 etaT;
	vec3 k;
};

void Init(out FresnelConductor fresnel, vec3 etaI, vec3 etaT, vec3 k)
{
	fresnel.etaI = etaI;
	fresnel.etaT = etaT;
	fresnel.k    = k;
}

struct FresnelDielectric
{
	float etaI;
	float etaT;
};

void Init(out FresnelDielectric fresnel, float etaI, float etaT)
{
	fresnel.etaI = etaI;
	fresnel.etaT = etaT;
}

// FresnelOp

vec3 FresnelEvaluate(FresnelConductor fresnel, float cosThetaI)
{
	cosThetaI = clamp(cosThetaI, -1.0, 1.0);

	vec3 eta  = fresnel.etaT / fresnel.etaI;
	vec3 etak = fresnel.k / fresnel.etaI;

	float cosThetaI2 = cosThetaI * cosThetaI;
	float sinThetaI2 = 1.0 - cosThetaI2;
	vec3  eta2       = eta * eta;
	vec3  etak2      = etak * etak;

	vec3 t0       = eta2 - etak2 - sinThetaI2;
	vec3 a2plusb2 = sqrt(t0 * t0 + 4.0 * eta2 * etak2);
	vec3 t1       = a2plusb2 + cosThetaI2;
	vec3 a        = sqrt(0.5 * (a2plusb2 + t0));
	vec3 t2       = 2.0 * cosThetaI * a;
	vec3 Rs       = (t1 - t2) / (t1 + t2);

	vec3 t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
	vec3 t4 = t2 * sinThetaI2;
	vec3 Rp = Rs * (t3 - t4) / (t3 + t4);

	return 0.5 * (Rp + Rs);
}

vec3 FresnelEvaluate(FresnelDielectric fresnel, float cosThetaI)
{
	cosThetaI = clamp(cosThetaI, -1.0, 1.0);

	// Potentially swap indices of refraction
	if (cosThetaI <= 0.f)
	{
		// Swap
		float temp   = fresnel.etaI;
		fresnel.etaI = fresnel.etaT;
		fresnel.etaT = temp;
		cosThetaI    = abs(cosThetaI);
	}

	float sinThetaI = sqrt(max(0.0, 1.0 - cosThetaI * cosThetaI));
	float sinThetaT = fresnel.etaI / fresnel.etaT * sinThetaI;
	if (sinThetaT >= 1.0)
	{
		return vec3(1.0);
	}

	float cosThetaT = sqrt(max(0.0, 1.0 - sinThetaT * sinThetaT));
	float Rparl     = ((fresnel.etaT * cosThetaI) - (fresnel.etaI * cosThetaT)) /
	              ((fresnel.etaT * cosThetaI) + (fresnel.etaI * cosThetaT));
	float Rperp = ((fresnel.etaI * cosThetaI) - (fresnel.etaT * cosThetaT)) /
	              ((fresnel.etaI * cosThetaI) + (fresnel.etaT * cosThetaT));

	return vec3(Rparl * Rparl + Rperp * Rperp) / 2;
}

// Evaluate FresnelOp
vec3 FresnelEvaluate()
{
	return vec3(1.0);
}

////////////// Lambertian Reflection //////////////
struct LambertianReflection
{
	vec3 R;
};

void Init(out LambertianReflection bxdf, vec3 base_color)
{
	bxdf.R = base_color;
}

vec3 Distribution(LambertianReflection bxdf, vec3 wo, vec3 wi)
{
	return bxdf.R * InvPI;
}

float Pdf(LambertianReflection bxdf, vec3 wo, vec3 wi)
{
	return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPI : 0.0;
}

vec3 SampleDistribution(in LambertianReflection bxdf, in vec3 wo, inout uint seed, out vec3 wi, out float pdf)
{
	wi = SampleCosineHemisphere(rand2(seed));
	if (wo.z < 0.0)
	{
		wi.z *= -1.0;
	}
	pdf = Pdf(bxdf, wo, wi);
	return Distribution(bxdf, wo, wi);
}

////////////// OrenNayar Reflection //////////////
struct OrenNayar
{
	vec3  R;
	float A, B;
};

void Init(out OrenNayar bxdf, vec3 R, float sigma)
{
	bxdf.R       = R;
	sigma        = Radians(sigma);
	float sigma2 = sigma * sigma;
	bxdf.A       = 1.0 - sigma2 / (2.0 * (sigma2 + 0.33));
	bxdf.B       = 0.45 * sigma2 / (sigma2 + 0.09);
}

vec3 Distribution(OrenNayar bxdf, vec3 wo, vec3 wi)
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
		maxCos     = max(0.0, dCos);
	}

	float tanThetaI = sinThetaI / AbsCosTheta(wi);
	float tanThetaO = sinThetaO / AbsCosTheta(wo);

	float sinAlpha, tanBeta;
	if (AbsCosTheta(wi) > AbsCosTheta(wo))
	{
		sinAlpha = sinThetaO;
		tanBeta  = sinThetaI / AbsCosTheta(wi);
	}
	else
	{
		sinAlpha = sinThetaI;
		tanBeta  = sinThetaO / AbsCosTheta(wo);
	}
	return bxdf.R * InvPI * (bxdf.A + bxdf.B * maxCos * sinAlpha * tanBeta);
}

float Pdf(OrenNayar bxdf, vec3 wo, vec3 wi)
{
	return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPI : 0.0;
}

vec3 SampleDistribution(in OrenNayar bxdf, in vec3 wo, inout uint seed, out vec3 wi, out float pdf)
{
	wi = SampleCosineHemisphere(rand2(seed));
	if (wo.z < 0.0)
	{
		wi.z *= -1.0;
	}
	pdf = Pdf(bxdf, wo, wi);
	return Distribution(bxdf, wo, wi);
}

////////////// Beckmann Distribution //////////////
struct BeckmannDistribution
{
	float alpha_x;
	float alpha_y;
	bool  sample_visible_area;
};

void Init(out BeckmannDistribution distribution, float alpha_x, float alpha_y, bool vis)
{
	distribution.alpha_x             = alpha_x;
	distribution.alpha_y             = alpha_y;
	distribution.sample_visible_area = vis;
}

float Distribution(BeckmannDistribution distribution, vec3 wh)
{
	float tan2Theta = Tan2Theta(wh);
	if (isinf(tan2Theta))
	{
		return 0.0;
	}

	float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
	return exp(-tan2Theta * (Cos2Phi(wh) / (distribution.alpha_x * distribution.alpha_x) +
	                         Sin2Phi(wh) / (distribution.alpha_y * distribution.alpha_y))) /
	       (PI * distribution.alpha_x * distribution.alpha_y * cos4Theta);
}

vec3 SampleWh(BeckmannDistribution distribution, vec3 wo, inout uint seed)
{
	vec2 u = rand2(seed);

	if (!distribution.sample_visible_area)
	{
		float tan2Theta, phi;
		if (distribution.alpha_x == distribution.alpha_y)
		{
			float log_sample = log(1 - u.x);
			tan2Theta        = -distribution.alpha_x * distribution.alpha_x * log_sample;
			phi              = u.y * 2 * PI;
		}
		else
		{
			float log_sample = log(1 - u.x);
			phi              = atan(distribution.alpha_y / distribution.alpha_x * tan(2.0 * PI * u.y + 0.5 * PI));
			if (u.y > 0.5)
			{
				phi += PI;
			}

			float sinPhi  = sin(phi);
			float cosPhi  = cos(phi);
			float alphax2 = distribution.alpha_x * distribution.alpha_x;
			float alphay2 = distribution.alpha_y * distribution.alpha_y;
			tan2Theta     = -log_sample / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
		}

		float cosTheta = 1.0 / sqrt(1.0 + tan2Theta);
		float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
		vec3  wh       = SphericalDirection(sinTheta, cosTheta, phi);
		if (!SameHemisphere(wo, wh))
		{
			wh = -wh;
		}
		return wh;
	}
	else
	{
		vec3 wh;
		bool flip = wo.z < 0.0;
		wh        = BeckmannSample(flip ? -wo : wo, distribution.alpha_x, distribution.alpha_y, u.x, u.y);
		if (flip)
		{
			wh = -wh;
		}

		return wh;
	}

	return vec3(0.0);
}

float Lambda(BeckmannDistribution distribution, vec3 w)
{
	float absTanTheta = abs(TanTheta(w));
	if (isinf(absTanTheta))
	{
		return 0.;
	}

	float alpha = sqrt(Cos2Phi(w) * distribution.alpha_x * distribution.alpha_x + Sin2Phi(w) * distribution.alpha_y * distribution.alpha_y);
	float a     = 1.0 / (alpha * absTanTheta);
	if (a >= 1.6)
	{
		return 0;
	}
	return (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a);
}

float G1(BeckmannDistribution distribution, vec3 w)
{
	return 1.0 / (1.0 + Lambda(distribution, w));
}

float G(BeckmannDistribution distribution, vec3 wo, vec3 wi)
{
	return 1.0 / (1.0 + Lambda(distribution, wo) + Lambda(distribution, wi));
}

float Pdf(BeckmannDistribution distribution, vec3 wo, vec3 wh)
{
	if (distribution.sample_visible_area)
	{
		return Distribution(distribution, wh) * G1(distribution, wo) * abs(dot(wo, wh)) / AbsCosTheta(wo);
	}
	else
	{
		return Distribution(distribution, wh) * AbsCosTheta(wh);
	}
}

////////////// Trowbridge Reitz Distribution //////////////
struct TrowbridgeReitzDistribution
{
	float alpha_x;
	float alpha_y;
	bool  sample_visible_area;
};

void Init(out TrowbridgeReitzDistribution distribution, float alpha_x, float alpha_y, bool vis)
{
	distribution.alpha_x             = alpha_x;
	distribution.alpha_y             = alpha_y;
	distribution.sample_visible_area = vis;
}

float Distribution(TrowbridgeReitzDistribution distribution, vec3 wh)
{
	float tan2Theta = Tan2Theta(wh);
	if (isinf(tan2Theta))
	{
		return 0.0;
	}
	const float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

	//return 1 / (PI * distribution.alpha_x * distribution.alpha_y * cos4Theta);

	float e = (Cos2Phi(wh) / (distribution.alpha_x * distribution.alpha_x) + Sin2Phi(wh) / (distribution.alpha_y * distribution.alpha_y)) * tan2Theta;
	return 1 / (PI * distribution.alpha_x * distribution.alpha_y * cos4Theta * (1 + e) * (1 + e));
}

vec3 SampleWh(TrowbridgeReitzDistribution distribution, vec3 wo, inout uint seed)
{
	vec3 wh = vec3(0.0);
	vec2 u  = rand2(seed);

	if (!distribution.sample_visible_area)
	{
		float cosTheta = 0.0;
		float phi      = 2.0 * PI * u.y;

		if (distribution.alpha_x == distribution.alpha_y)
		{
			float tanTheta2 = distribution.alpha_x * distribution.alpha_x * u.x * (1.0 - u.y);
			cosTheta        = 1.0 / sqrt(1.0 + tanTheta2);
		}
		else
		{
			phi = atan(distribution.alpha_y / distribution.alpha_x * tan(2.0 * PI * u.y + 0.5 * PI));
			if (u.y > 0.5)
			{
				phi += PI;
			}
			float sinPhi = sin(phi);
			float cosPhi = cos(phi);

			const float alpha_x2 = distribution.alpha_x * distribution.alpha_x;
			const float alpha_y2 = distribution.alpha_y * distribution.alpha_y;

			const float alpha_2   = 1.0 / (cosPhi * cosPhi / alpha_x2 + sinPhi * sinPhi / alpha_y2);
			float       tanTheta2 = alpha_2 * u.x / (1.0 - u.x);
			cosTheta              = 1.0 / sqrt(1 + tanTheta2);
		}

		float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
		vec3  wh       = SphericalDirection(sinTheta, cosTheta, phi);
		if (!SameHemisphere(wo, wh))
		{
			wh = -wh;
		}
	}
	else
	{
		bool flip = wo.z < 0.0;
		wh        = TrowbridgeReitzSample(flip ? -wo : wo, distribution.alpha_x, distribution.alpha_y, u.x, u.y);
		if (flip)
		{
			wh = -wh;
		}
	}
	return wh;
}

float Lambda(TrowbridgeReitzDistribution distribution, vec3 w)
{
	float absTanTheta = abs(TanTheta(w));
	if (isinf(absTanTheta))
	{
		return 0.;
	}
	float alpha           = sqrt(Cos2Phi(w) * distribution.alpha_x * distribution.alpha_x + Sin2Phi(w) * distribution.alpha_y * distribution.alpha_y);
	float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
	return (-1.0 + sqrt(1.0 + alpha2Tan2Theta)) / 2.0;
}

float G1(TrowbridgeReitzDistribution distribution, vec3 w)
{
	return 1.0 / (1.0 + Lambda(distribution, w));
}

float G(TrowbridgeReitzDistribution distribution, vec3 wo, vec3 wi)
{
	return 1.0 / (1.0 + Lambda(distribution, wo) + Lambda(distribution, wi));
}

float Pdf(TrowbridgeReitzDistribution distribution, vec3 wo, vec3 wh)
{
	if (distribution.sample_visible_area)
	{
		return Distribution(distribution, wh) * G1(distribution, wo) * abs(dot(wo, wh)) / AbsCosTheta(wo);
	}
	else
	{
		return Distribution(distribution, wh) * AbsCosTheta(wh);
	}
}

////////////// Microfacet Reflection //////////////
struct MicrofacetReflection
{
	vec3 R;
};

void Init(out MicrofacetReflection bxdf, vec3 base_color)
{
	bxdf.R = base_color;
}

vec3 Distribution(MicrofacetReflection bxdf, vec3 F, float D, float G, vec3 wo, vec3 wi)
{
	float cosThetaO = AbsCosTheta(wo);
	float cosThetaI = AbsCosTheta(wi);

	vec3 wh = wi + wo;

	if (cosThetaI == 0.0 || cosThetaO == 0.0)
	{
		return vec3(0.0);
	}
	if (wh.x == 0.0 && wh.y == 0.0 && wh.z == 0.0)
	{
		return vec3(0.0);
	}

	return bxdf.R * D * G * F / (4.0 * cosThetaI * cosThetaO);
}

float Pdf(MicrofacetReflection bxdf, vec3 wo, vec3 wi, vec3 wh, float distribution_pdf)
{
	if (!SameHemisphere(wo, wi))
	{
		return 0;
	}

	return distribution_pdf / (4.0 * dot(wo, wh));
}

vec3 SampleDistribution(in MicrofacetReflection bxdf, in vec3 wo, in vec3 wh, in vec3 F, in float D, in float G, in float distribution_pdf, inout uint seed, out vec3 wi, out float pdf)
{
	if (wo.z < 0.0)
	{
		return vec3(0.0);
	}

	if (dot(wo, wh) < 0.0)
	{
		return vec3(0.0);
	}

	wi = reflect(wo, wh);

	if (!SameHemisphere(wo, wi))
	{
		return vec3(0.0);
	}

	pdf = distribution_pdf / (4.0 * dot(wo, wh));
	return Distribution(bxdf, F, D, G, wo, wi);
}

////////////// Specular Reflection //////////////
struct SpecularReflection
{
	vec3 R;
};

void Init(out SpecularReflection bxdf, vec3 base_color)
{
	bxdf.R = base_color;
}

vec3 Distribution(SpecularReflection bxdf, vec3 F, vec3 wo, vec3 wi)
{
	return vec3(0.0);
}

float Pdf(SpecularReflection bxdf, vec3 wo, vec3 wi)
{
	return 0.0;
}

vec3 SampleDistribution(in SpecularReflection bxdf, in vec3 wo, in vec3 F, inout uint seed, out vec3 wi, out float pdf)
{
	wi  = vec3(-wo.x, -wo.y, wo.z);
	pdf = 1.0;
	return F * bxdf.R / AbsCosTheta(wi);
}

////////////// Fresnel Blend //////////////
struct FresnelBlend
{
	vec3 Rd;
	vec3 Rs;
};

void Init(out FresnelBlend bxdf, vec3 Rd, vec3 Rs)
{
	bxdf.Rd = Rd;
	bxdf.Rs = Rs;
}

vec3 SchlickFresnel(FresnelBlend bxdf, float cosTheta)
{
	return bxdf.Rs + pow(1 - cosTheta, 5.0) * (vec3(1.0) - bxdf.Rs);
}

vec3 Distribution(FresnelBlend bxdf, TrowbridgeReitzDistribution distribution, vec3 wo, vec3 wi)
{
	vec3 diffuse = (28.0 / (23.0 * PI)) * bxdf.Rd * (1.0 - bxdf.Rs) *
	               (1.0 - pow(1.0 - 0.5 * AbsCosTheta(wi), 5.0)) *
	               (1.0 - pow(1.0 - 0.5 * AbsCosTheta(wo), 5.0));

	vec3 wh = wi + wo;

	if (wh.x == 0.0 && wh.y == 0.0 && wh.z == 0.0)
	{
		return vec3(0.0);
	}
	wh = normalize(wh);

	float D = Distribution(distribution, wh);

	vec3 specular = D / (4.0 * abs(dot(wi, wh) * max(AbsCosTheta(wi), AbsCosTheta(wo)))) *
	                SchlickFresnel(bxdf, dot(wi, wh));

	return diffuse + specular;
}

vec3 Distribution(FresnelBlend bxdf, BeckmannDistribution distribution, vec3 wo, vec3 wi)
{
	vec3 diffuse = (28.0 / (23.0 * PI)) * bxdf.Rd * (1.0 - bxdf.Rs) *
	               (1.0 - pow(1.0 - 0.5 * AbsCosTheta(wi), 5.0)) *
	               (1.0 - pow(1.0 - 0.5 * AbsCosTheta(wo), 5.0));

	vec3 wh = wi + wo;

	if (wh.x == 0.0 && wh.y == 0.0 && wh.z == 0.0)
	{
		return vec3(0.0);
	}
	wh = normalize(wh);

	float D = Distribution(distribution, wh);

	vec3 specular = D / (4.0 * abs(dot(wi, wh) * max(AbsCosTheta(wi), AbsCosTheta(wo)))) *
	                SchlickFresnel(bxdf, dot(wi, wh));

	return diffuse + specular;
}

float Pdf(FresnelBlend bxdf, TrowbridgeReitzDistribution distribution, vec3 wo, vec3 wi)
{
	if (!SameHemisphere(wo, wi))
	{
		return 0.0;
	}

	vec3 wh = normalize(wo + wi);
	float pdf_wh = Pdf(distribution, wo, wh);
	return 0.5 * (AbsCosTheta(wi) * InvPI + pdf_wh / (4.0 * dot(wo, wh)));
}

float Pdf(FresnelBlend bxdf, BeckmannDistribution distribution, vec3 wo, vec3 wi)
{
	if (!SameHemisphere(wo, wi))
	{
		return 0.0;
	}

	vec3  wh     = normalize(wo + wi);
	float pdf_wh = Pdf(distribution, wo, wh);
	return 0.5 * (AbsCosTheta(wi) * InvPI + pdf_wh / (4.0 * dot(wo, wh)));
}

vec3 SampleDistribution(in FresnelBlend bxdf, in vec3 wo, in TrowbridgeReitzDistribution distribution, inout uint seed, out vec3 wi, out float pdf)
{
	vec2 u = rand2(seed);

	if (u.x < 0.5)
	{
		u.x = min(2.0 * u.x, 0.999999);
		wi  = SampleCosineHemisphere(u);
		if (wo.z < 0.0)
		{
			wi.z *= -1.0;
		}
	}
	else
	{
		u.x = min(2.0 * (u.x - 0.5), 0.999999);
		vec3 wh = SampleWh(distribution, wo, seed);
		wi      = reflect(wo, wh);
		if (!SameHemisphere(wo, wi))
		{
			return vec3(0.0);
		}
	}

	pdf = Pdf(bxdf, distribution, wo, wi);
	return Distribution(bxdf, distribution, wo, wi);
}

vec3 SampleDistribution(in FresnelBlend bxdf, in vec3 wo, in BeckmannDistribution distribution, inout uint seed, out vec3 wi, out float pdf)
{
	vec2 u = rand2(seed);

	if (u.x < 0.5)
	{
		u.x = min(2.0 * u.x, 0.999999);
		wi  = SampleCosineHemisphere(u);
		if (wo.z < 0.0)
		{
			wi.z *= -1.0;
		}
	}
	else
	{
		u.x     = min(2.0 * (u.x - 0.5), 0.999999);
		vec3 wh = SampleWh(distribution, wo, seed);
		wi      = reflect(wo, wh);
		if (!SameHemisphere(wo, wi))
		{
			return vec3(0.0);
		}
	}

	pdf = Pdf(bxdf, distribution, wo, wi);
	return Distribution(bxdf, distribution, wo, wi);
}
#endif