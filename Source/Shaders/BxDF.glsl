#ifndef _BXDF_GLSL_
#define _BXDF_GLSL_

#define PI 3.141592653589793
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

#endif