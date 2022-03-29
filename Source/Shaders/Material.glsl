#ifndef _MATERIAL_H
#define _MATERIAL_H

#include "BxDf.glsl"
#include "GlobalBuffer.glsl"

////////////// Matte MaterialData //////////////
struct MatteMaterial
{
	vec3  Kd;
	float sigma;
};

void Init(out MatteMaterial mat, vec3 base_color, float roughness)
{
	mat.Kd    = base_color;
	mat.sigma = roughness;
}

vec3 Distribution(MatteMaterial mat, vec3 wo, vec3 wi)
{
	if (wo.z == 0.0)
	{
		return vec3(0.0);
	}

	vec3  r   = mat.Kd;
	float sig = clamp(mat.sigma, 0.0, 90.0);
	if (sig == 0)
	{
		// Use Lambertian Reflection
		LambertianReflection bxdf;
		Init(bxdf, r);
		return Distribution(bxdf, wo, wi);
	}
	else
	{
		// Use Oren Nayar Reflection
		OrenNayar bxdf;
		Init(bxdf, mat.Kd, mat.sigma);
		return Distribution(bxdf, wo, wi);
	}
}

vec3 SampleDistribution(in MatteMaterial mat, in vec3 wo, inout uint seed, out vec3 wi, out float pdf)
{
	if (wo.z == 0.0)
	{
		return vec3(0.0);
	}

	vec3  r   = mat.Kd;
	float sig = clamp(mat.sigma, 0.0, 90.0);
	if (sig == 0)
	{
		// Use Lambertian Reflection
		LambertianReflection bxdf;
		Init(bxdf, r);
		return SampleDistribution(bxdf, wo, seed, wi, pdf);
	}
	else
	{
		// Use Oren Nayar Reflection
		OrenNayar bxdf;
		Init(bxdf, mat.Kd, mat.sigma);
		return SampleDistribution(bxdf, wo, seed, wi, pdf);
	}
}

////////////// Plastic MaterialData //////////////
struct PlasticMaterial
{
	vec3  Kd;
	vec3  Ks;
	float roughness;
};

void Init(out PlasticMaterial mat, vec3 diffuse, vec3 specular, float roughness)
{
	mat.Kd        = diffuse;
	mat.Ks        = specular;
	mat.roughness = roughness;
}

vec3 Distribution(PlasticMaterial mat, vec3 wo, vec3 wi)
{
	vec3 distribution = vec3(0.0);

	// Diffuse term
	if (mat.Kd != vec3(0.0))
	{
		LambertianReflection bxdf;
		Init(bxdf, mat.Kd);
		distribution += Distribution(bxdf, wo, wi);
	}

	// Specular term
	if (mat.Ks != vec3(0.0))
	{
		TrowbridgeReitzDistribution dist;
		float rough = RoughnessToAlpha(mat.roughness);
		Init(dist, rough, rough, false);

		MicrofacetReflection bxdf;
		Init(bxdf, mat.Ks);

		FresnelDielectric fresnel;
		Init(fresnel, 1.0, 1.5);

		vec3 wh = normalize(wi + wo);

		vec3 F = FresnelEvaluate(fresnel, dot(wi, Faceforward(wh, vec3(0.0, 0.0, 1.0))));
		float D = Distribution(dist, wh);
		float G = G(dist, wo, wi);

		distribution += Distribution(bxdf, F, D, G, wo, wi);
	}

	return distribution;
}

vec3 SampleDistribution(in PlasticMaterial mat, in vec3 wo, inout uint seed, out vec3 wi, out float pdf)
{
	vec3 distribution = vec3(0.0);

	// Random Select one bxdf to sample
	float u = rand(seed);

	if (u < 0.5)
	{
		// Choose Lambertian term
		LambertianReflection bxdf;
		Init(bxdf, mat.Kd);
		return SampleDistribution(bxdf, wo, seed, wi, pdf);
	}
	else
	{
		// Choose microfacet term
		TrowbridgeReitzDistribution dist;
		float                       rough = RoughnessToAlpha(mat.roughness);
		Init(dist, rough, rough, false);

		MicrofacetReflection bxdf;
		Init(bxdf, mat.Ks);

		FresnelDielectric fresnel;
		Init(fresnel, 1.5, 1.0);

		vec3 wh = normalize(wi + wo);

		vec3  F = FresnelEvaluate(fresnel, dot(wi, Faceforward(wh, vec3(0.0, 0.0, 1.0))));
		float D = Distribution(dist, wh);
		float G = G(dist, wo, wi);

		return SampleDistribution(bxdf, wo, wh, F, D, G, Pdf(dist, wo, wh), seed, wi, pdf);
	}
}

////////////// Material Sampling//////////////
vec3 Distribution(Material mat, vec3 wo, vec3 wi)
{
	if (mat.material_type == BxDF_Matte)
	{
		MatteMaterial matte;
		Init(matte, mat.base_color.rgb, mat.roughness);
		return Distribution(matte, wo, wi);
	}
	else if (mat.material_type == BxDF_Plastic)
	{
		PlasticMaterial plastic;
		Init(plastic, mat.base_color.rgb, vec3(mat.specular), mat.roughness);
		return Distribution(plastic, wo, wi);
	}

	return vec3(0.0);
}

vec3 SampleDistribution(in Material mat, in vec3 wo, inout uint seed, out vec3 wi, out float pdf)
{
	if (mat.material_type == BxDF_Matte)
	{
		MatteMaterial matte;
		Init(matte, mat.base_color.rgb, mat.roughness);
		return SampleDistribution(matte, wo, seed, wi, pdf);
	}
	else if (mat.material_type == BxDF_Plastic)
	{
		PlasticMaterial plastic;
		Init(plastic, mat.base_color.rgb, vec3(mat.specular), mat.roughness);
		return SampleDistribution(plastic, wo, seed, wi, pdf);
	}

	return vec3(0.0);
}

#endif