#ifndef _MATERIAL_H
#define _MATERIAL_H

#include "GlobalBuffer.glsl"

////////////// Matte Material //////////////
struct MatteMaterial
{
	vec3 Kd;
	float sigma;
};

void Init(out MatteMaterial mat, vec3 base_color, float roughness)
{
	mat.Kd    = base_color;
	mat.sigma = roughness * 90.0;
}

vec3 Distribution(MatteMaterial mat, vec3 wo, vec3 wi)
{
	if (wo.z == 0.0)
	{
		return vec3(0.0);
	}

	vec3 r = mat.Kd;
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

vec3 SampleDistribution(in MatteMaterial mat, in vec3 wo, uint seed, out vec3 wi, out float pdf)
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

#endif