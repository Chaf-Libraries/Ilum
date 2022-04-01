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
		float                       rough = (mat.roughness);
		Init(dist, rough, rough, true);

		MicrofacetReflection bxdf;
		Init(bxdf, mat.Ks);

		FresnelDielectric fresnel;
		Init(fresnel, 1.0, 1.5);

		distribution += Distribution(bxdf, fresnel, dist, wo, wi);
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
		float                       rough = (mat.roughness);
		Init(dist, rough, rough, true);

		MicrofacetReflection bxdf;
		Init(bxdf, mat.Ks);

		FresnelDielectric fresnel;
		Init(fresnel, 1.0, 1.5);

		return SampleDistribution(bxdf, wo, fresnel, dist, seed, wi, pdf);
	}
}

////////////// Metal MaterialData //////////////
struct MetalMaterial
{
	vec3  R;
	vec3  eta;
	vec3  k;
	float anisotropic;
	float roughness;
};

void Init(out MetalMaterial mat, vec3 base_color, float anisotropic, float roughness)
{
	mat.R           = base_color;
	mat.eta         = vec3(1, 10, 11);
	mat.k           = vec3(3.90463543, 2.44763327, 2.13765264);
	mat.anisotropic = anisotropic;
	mat.roughness   = roughness;
}

vec3 Distribution(MetalMaterial mat, vec3 wo, vec3 wi)
{
	float urough = max(mat.roughness * (1.0 + mat.anisotropic), 0.00001);
	float vrough = max(mat.roughness * (1.0 - mat.anisotropic), 0.00001);

	urough = urough;
	vrough = vrough;

	FresnelConductor fresnel;
	Init(fresnel, vec3(1.0), mat.eta, mat.k);

	TrowbridgeReitzDistribution dist;
	Init(dist, urough, vrough, true);

	MicrofacetReflection bxdf;
	Init(bxdf, mat.R);

	return Distribution(bxdf, fresnel, dist, wo, wi);
}

vec3 SampleDistribution(in MetalMaterial mat, in vec3 wo, inout uint seed, out vec3 wi, out float pdf)
{
	float urough = max(mat.roughness * (1.0 + mat.anisotropic), 0.00001);
	float vrough = max(mat.roughness * (1.0 - mat.anisotropic), 0.00001);

	urough = urough;
	vrough = vrough;

	FresnelConductor fresnel;
	Init(fresnel, vec3(1.0), mat.eta, mat.k);

	TrowbridgeReitzDistribution dist;
	Init(dist, urough, vrough, true);

	MicrofacetReflection bxdf;
	Init(bxdf, mat.R);

	return SampleDistribution(bxdf, wo, fresnel, dist, seed, wi, pdf);
}

////////////// Mirror MaterialData //////////////
struct MirrorMaterial
{
	vec3 R;
};

void Init(out MirrorMaterial mat, vec3 base_color)
{
	mat.R = base_color;
}

vec3 Distribution(MirrorMaterial mat, vec3 wo, vec3 wi)
{
	SpecularReflection bxdf;
	Init(bxdf, mat.R);

	return Distribution(bxdf, wo, wi);
}

vec3 SampleDistribution(in MirrorMaterial mat, in vec3 wo, inout uint seed, out vec3 wi, out float pdf)
{
	SpecularReflection bxdf;
	Init(bxdf, mat.R);

	return SampleDistribution(bxdf, wo, seed, wi, pdf);
}

////////////// Substrate MaterialData //////////////
struct SubstrateMaterial
{
	vec3  Kd;
	vec3  Rs;
	float anisotropic;
	float roughness;
};

void Init(out SubstrateMaterial mat, vec3 base_color, vec3 glossy, float anisotropic, float roughness)
{
	mat.Kd          = base_color;
	mat.Rs          = glossy;
	mat.anisotropic = anisotropic;
	mat.roughness   = roughness;
}

vec3 Distribution(SubstrateMaterial mat, vec3 wo, vec3 wi)
{
	FresnelBlend bxdf;
	Init(bxdf, mat.Kd, vec3(mat.Rs));

	float urough = max(mat.roughness * (1.0 + mat.anisotropic), 0.00001);
	float vrough = max(mat.roughness * (1.0 - mat.anisotropic), 0.00001);

	urough = urough;
	vrough = vrough;

	TrowbridgeReitzDistribution dist;
	Init(dist, urough, vrough, true);

	return Distribution(bxdf, dist, wo, wi);
}

vec3 SampleDistribution(in SubstrateMaterial mat, in vec3 wo, inout uint seed, out vec3 wi, out float pdf)
{
	FresnelBlend bxdf;
	Init(bxdf, mat.Kd, vec3(mat.Rs));

	float urough = max(mat.roughness * (1.0 + mat.anisotropic), 0.00001);
	float vrough = max(mat.roughness * (1.0 - mat.anisotropic), 0.00001);

	urough = urough;
	vrough = vrough;

	TrowbridgeReitzDistribution dist;
	Init(dist, urough, vrough, true);

	return SampleDistribution(bxdf, wo, dist, seed, wi, pdf);
}

////////////// Glass MaterialData //////////////
struct GlassMaterial
{
	float refraction;
	vec3  R;
	vec3  T;
	float anisotropic;
	float roughness;
};

void Init(out GlassMaterial mat, vec3 reflection_color, vec3 transmission_color, float anisotropic, float roughness, float refraction)
{
	mat.refraction  = refraction;
	mat.R           = reflection_color;
	mat.T           = transmission_color;
	mat.anisotropic = anisotropic;
	mat.roughness   = roughness;
}

vec3 Distribution(GlassMaterial mat, vec3 wo, vec3 wi)
{
	vec3 distribution = vec3(0.0);

	if (mat.R == vec3(0.0) && mat.T == vec3(0.0))
	{
		return vec3(0.0);
	}

	bool isSpecular = (mat.roughness == 0.0);

	float urough = max(mat.roughness * (1.0 + mat.anisotropic), 0.00001);
	float vrough = max(mat.roughness * (1.0 - mat.anisotropic), 0.00001);

	urough = urough;
	vrough = vrough;

	TrowbridgeReitzDistribution dist;
	Init(dist, urough, vrough, true);

	FresnelDielectric fresnel;
	Init(fresnel, 1.0, mat.refraction);

	if (isSpecular)
	{
		FresnelSpecular bxdf;
		Init(bxdf, mat.R, mat.T, 1.0, mat.refraction);
		distribution += Distribution(bxdf, wo, wi);
	}
	else
	{
		if (mat.R != vec3(0.0))
		{
			MicrofacetReflection bxdf;
			Init(bxdf, mat.R);

			distribution += Distribution(bxdf, fresnel, dist, wo, wi);
		}
		if (mat.T != vec3(0.0))
		{
			MicrofacetTransmission bxdf;
			Init(bxdf, mat.T, 1.0, mat.refraction);

			distribution += Distribution(bxdf, fresnel, dist, wo, wi);
		}
	}

	return distribution;
}

vec3 SampleDistribution(in GlassMaterial mat, in vec3 wo, inout uint seed, out vec3 wi, out float pdf)
{
	if (mat.R == vec3(0.0) && mat.T == vec3(0.0))
	{
		return vec3(0.0);
	}

	vec3 distribution = vec3(0.0);

	float u = rand(seed);

	bool isSpecular = (mat.roughness == 0.0);

	float urough = max(mat.roughness * (1.0 + mat.anisotropic), 0.00001);
	float vrough = max(mat.roughness * (1.0 - mat.anisotropic), 0.00001);

	urough = urough;
	vrough = vrough;

	TrowbridgeReitzDistribution dist;
	Init(dist, urough, vrough, true);

	FresnelDielectric fresnel;
	Init(fresnel, 1.0, mat.refraction);

	if (isSpecular)
	{
		FresnelSpecular bxdf;
		Init(bxdf, mat.R, mat.T, 1.0, mat.refraction);
		distribution = SampleDistribution(bxdf, wo, seed, wi, pdf);
		return distribution;
	}
	else
	{
		if (mat.R != vec3(0.0) && mat.T == vec3(0.0))
		{
			MicrofacetReflection bxdf;
			Init(bxdf, mat.R);
			distribution = SampleDistribution(bxdf, wo, fresnel, dist, seed, wi, pdf);
			return distribution;
		}
		else if (mat.R == vec3(0.0) && mat.T != vec3(0.0))
		{
			MicrofacetTransmission bxdf;
			Init(bxdf, mat.T, 1.0, mat.refraction);
			distribution = SampleDistribution(bxdf, wo, fresnel, dist, seed, wi, pdf);
			return distribution;
		}
		else
		{
			float eta = CosTheta(wo) > 0.0 ? (mat.refraction / 1.0) : (1.0 / mat.refraction);
			vec3  wh  = normalize(wo + wi * eta);
			vec3  F   = FresnelEvaluate(fresnel, dot(wo, wh));

			if (u.x < F.x)
			{
				// Reflection
				MicrofacetReflection bxdf;
				Init(bxdf, mat.R);
				distribution = SampleDistribution(bxdf, wo, fresnel, dist, seed, wi, pdf);
				pdf *= F.x;
				return distribution;
			}
			else
			{
				// Refraction
				MicrofacetTransmission bxdf;
				Init(bxdf, mat.T, 1.0, mat.refraction);
				distribution = SampleDistribution(bxdf, wo, fresnel, dist, seed, wi, pdf);
				pdf *= 1.0 - F.x;
				return distribution;
			}
		}
	}

	return distribution;
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
	else if (mat.material_type == BxDF_Metal)
	{
		MetalMaterial metal;
		Init(metal, mat.base_color.rgb, mat.anisotropic, mat.roughness);
		return Distribution(metal, wo, wi);
	}
	else if (mat.material_type == BxDF_Mirror)
	{
		MirrorMaterial mirror;
		Init(mirror, mat.base_color.rgb);
		return Distribution(mirror, wo, wi);
	}
	else if (mat.material_type == BxDF_Substrate)
	{
		SubstrateMaterial substrate;
		Init(substrate, mat.base_color.rgb, mat.data, mat.anisotropic, mat.roughness);
		return Distribution(substrate, wo, wi);
	}
	else if (mat.material_type == BxDF_Substrate)
	{
		GlassMaterial glass;
		Init(glass, mat.base_color.rgb, mat.data, mat.anisotropic, mat.roughness, mat.transmission);
		return Distribution(glass, wo, wi);
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
	else if (mat.material_type == BxDF_Metal)
	{
		MetalMaterial metal;
		Init(metal, mat.base_color.rgb, mat.anisotropic, mat.roughness);
		return SampleDistribution(metal, wo, seed, wi, pdf);
	}
	else if (mat.material_type == BxDF_Mirror)
	{
		MirrorMaterial mirror;
		Init(mirror, mat.base_color.rgb);
		return SampleDistribution(mirror, wo, seed, wi, pdf);
	}
	else if (mat.material_type == BxDF_Substrate)
	{
		SubstrateMaterial substrate;
		Init(substrate, mat.base_color.rgb, mat.data, mat.anisotropic, mat.roughness);
		return SampleDistribution(substrate, wo, seed, wi, pdf);
	}
	else if (mat.material_type == BxDF_Glass)
	{
		GlassMaterial glass;
		Init(glass, mat.base_color.rgb, mat.data, mat.anisotropic, mat.roughness, mat.transmission);
		return SampleDistribution(glass, wo, seed, wi, pdf);
	}

	return vec3(0.0);
}

#endif