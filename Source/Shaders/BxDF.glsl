#ifndef _BXDF_GLSL_
#define _BXDF_GLSL_

#define PI 3.1415926535

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
vec3 DisneyBRDF(vec3 L, vec3 V, vec3 N, vec3 X, vec3 Y,
                vec3 albedo, float metallic, float roughness, float anisotropic,
                float specular_, float specular_tint, float sheen, float sheen_tint,
                float clearcoat, float clearcoat_gloss, float subsurface)
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

	vec3 Cdlin = pow(albedo, vec3(2.2));
	// Luminace approximation
	float Cdlum = 0.3 * Cdlin.x + 0.1 * Cdlin.y + 0.1 * Cdlin.z;

	vec3 Ctint  = Cdlum > 0 ? Cdlin / Cdlum : vec3(1.0);
	vec3 Cspec0 = mix(specular_ * 0.08 * mix(vec3(1.0), Ctint, specular_tint), Cdlin, metallic);
	vec3 Csheen = mix(vec3(1.0), Ctint, sheen_tint);

	// Diffuse term
	float FL   = pow(clamp(1.0 - NoL, 0.0, 1.0), 5.0);
	float FV   = pow(clamp(1.0 - NoV, 0.0, 1.0), 5.0);
	float Fd90 = 0.5 + 2 * LoH * LoH * roughness;
	float Fd   = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);

	// Subsurface term
	// Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
	float Fss90 = LoH * LoH * roughness;
	float Fss   = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);
	float ss    = 1.25 * (Fss * (1.0 / (NoL + NoV) - 0.5) + 0.5);

	// Specular term
	float aspect = sqrt(1.0 - anisotropic * 0.9);
	float ax     = max(0.001, sqrt(roughness) / aspect);
	float ay     = max(0.001, sqrt(roughness) * aspect);
	//float HoX, float HoY, float NoH, float ax, float ay
	float Ds = DistributeAnisotropicGGX(dot(H, X), dot(H, Y), NoH, ax, ay);

	// Fresnel term
	vec3 Fs = FresnelSchlick(LoH, Cspec0);

	// Geometry term
	float Gs = GeometryAnisotropicSmithGGX(NoL, dot(L, X), dot(L, Y), ax, ay);
	Gs *= GeometryAnisotropicSmithGGX(NoV, dot(V, X), dot(V, Y), ax, ay);

	// Sheen
	float FH     = pow(clamp(1 - LoH, 0, 1), 5.0);
	vec3  Fsheen = FH * sheen * Csheen;

	// Clearcoat: ior = 1.5 -> F0 = 0.04
	float Dr = DistributeBerry(NoH, mix(0.1, 0.001, clearcoat_gloss));
	float Fr = mix(0.04, 1.0, FH);
	float Gr = GeometrySmithGGX(NoL, 0.25) * GeometrySmithGGX(NoV, 0.25);

	return ((1 / PI) * mix(Fd, ss, subsurface) * Cdlin + Fsheen) * (1 - metallic) + Gs * Fs * Ds + 0.25 * clearcoat * Gr * Fr * Dr;
}

// Cook Torrance BRDF
vec3 CookTorranceBRDF(vec3 L, vec3 V, vec3 N, vec3 X, vec3 Y,
                      vec3 albedo, float metallic, float roughness, float anisotropic)
{
	vec3 Cdlin = pow(albedo, vec3(2.2));
	vec3 F0 = vec3(0.04);
	F0         = mix(F0, Cdlin, metallic);

	vec3  H   = normalize(V + L);
	float NoV = clamp(dot(N, V), 0.0, 1.0);
	float NoL = clamp(dot(N, L), 0.0, 1.0);
	float NoH = clamp(dot(N, H), 0.0, 1.0);
	float HoV = clamp(dot(H, V), 0.0, 1.0);

	float D = DistributeGGX(NoH, roughness);
	float G = GeometrySmith(NoL, NoV, roughness);
	vec3  F = FresnelSchlick(HoV, F0);

	vec3 specular = D * F * G / (4.0 * NoL * NoV + 0.001);
	vec3 Kd       = (vec3(1.0) - F) * (1 - metallic);

	return Kd * LambertianDiffuse(Cdlin) + specular;
}

#endif