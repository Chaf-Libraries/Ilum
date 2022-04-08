#ifndef _MATERIAL_H
#define _MATERIAL_H

#include "BxDf.glsl"
#include "GlobalBuffer.glsl"

////////////// Matte Material //////////////
struct MatteMaterial
{
	vec3  Kd;
	float sigma;
};

void Init(out MatteMaterial mat, Material data)
{
	mat.Kd    = data.base_color.rgb;
	mat.sigma = data.roughness;
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

////////////// Plastic Material //////////////
struct PlasticMaterial
{
	vec3  Kd;
	vec3  Ks;
	float roughness;
};

void Init(out PlasticMaterial mat, Material data)
{
	mat.Kd        = data.base_color.rgb;
	mat.Ks        = data.data;
	mat.roughness = data.roughness;
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

////////////// Metal Material //////////////
struct MetalMaterial
{
	vec3  R;
	vec3  eta;
	vec3  k;
	float anisotropic;
	float roughness;
};

void Init(out MetalMaterial mat, Material data)
{
	mat.R           = data.base_color.rgb;
	mat.eta         = vec3(1, 10, 11);
	mat.k           = vec3(3.90463543, 2.44763327, 2.13765264);
	mat.anisotropic = data.anisotropic;
	mat.roughness   = data.roughness;
}

vec3 Distribution(MetalMaterial mat, vec3 wo, vec3 wi)
{
	float aspect = sqrt(1.0 - mat.anisotropic * 0.9);
	float urough = max(0.001, mat.roughness / aspect);
	float vrough = max(0.001, mat.roughness * aspect);

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
	float aspect = sqrt(1.0 - mat.anisotropic * 0.9);
	float urough = max(0.001, mat.roughness / aspect);
	float vrough = max(0.001, mat.roughness * aspect);

	FresnelConductor fresnel;
	Init(fresnel, vec3(1.0), mat.eta, mat.k);

	TrowbridgeReitzDistribution dist;
	Init(dist, urough, vrough, true);

	MicrofacetReflection bxdf;
	Init(bxdf, mat.R);

	return SampleDistribution(bxdf, wo, fresnel, dist, seed, wi, pdf);
}

////////////// Mirror Material //////////////
struct MirrorMaterial
{
	vec3 R;
};

void Init(out MirrorMaterial mat, Material data)
{
	mat.R = data.base_color.rgb;
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

////////////// Substrate Material //////////////
struct SubstrateMaterial
{
	vec3  Kd;
	vec3  Rs;
	float anisotropic;
	float roughness;
};

void Init(out SubstrateMaterial mat, Material data)
{
	mat.Kd          = data.base_color.rgb;
	mat.Rs          = data.data;
	mat.anisotropic = data.anisotropic;
	mat.roughness   = data.roughness;
}

vec3 Distribution(SubstrateMaterial mat, vec3 wo, vec3 wi)
{
	FresnelBlend bxdf;
	Init(bxdf, mat.Kd, vec3(mat.Rs));

	float aspect = sqrt(1.0 - mat.anisotropic * 0.9);
	float urough = max(0.001, mat.roughness / aspect);
	float vrough = max(0.001, mat.roughness * aspect);

	TrowbridgeReitzDistribution dist;
	Init(dist, urough, vrough, true);

	return Distribution(bxdf, dist, wo, wi);
}

vec3 SampleDistribution(in SubstrateMaterial mat, in vec3 wo, inout uint seed, out vec3 wi, out float pdf)
{
	FresnelBlend bxdf;
	Init(bxdf, mat.Kd, vec3(mat.Rs));

	float aspect = sqrt(1.0 - mat.anisotropic * 0.9);
	float urough = max(0.001, mat.roughness / aspect);
	float vrough = max(0.001, mat.roughness * aspect);

	TrowbridgeReitzDistribution dist;
	Init(dist, urough, vrough, true);

	return SampleDistribution(bxdf, wo, dist, seed, wi, pdf);
}

////////////// Glass Material //////////////
struct GlassMaterial
{
	float refraction;
	vec3  R;
	vec3  T;
	float anisotropic;
	float roughness;
};

void Init(out GlassMaterial mat, Material data)
{
	mat.refraction  = data.transmission;
	mat.R           = data.base_color.rgb;
	mat.T           = data.data;
	mat.anisotropic = data.anisotropic;
	mat.roughness   = data.roughness;
}

vec3 Distribution(GlassMaterial mat, vec3 wo, vec3 wi)
{
	vec3 distribution = vec3(0.0);

	if (mat.R == vec3(0.0) && mat.T == vec3(0.0))
	{
		return vec3(0.0);
	}

	bool isSpecular = (mat.roughness == 0.0);

	float aspect = sqrt(1.0 - mat.anisotropic * 0.9);
	float urough = max(0.001, mat.roughness / aspect);
	float vrough = max(0.001, mat.roughness * aspect);

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

	float aspect = sqrt(1.0 - mat.anisotropic * 0.9);
	float urough = max(0.001, mat.roughness / aspect);
	float vrough = max(0.001, mat.roughness * aspect);

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

////////////// Disney Material //////////////
struct DisneyMaterial
{
	vec3  color;
	float metallic;
	float refraction;
	float roughness;
	float specularTint;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatGloss;
	float specTrans;
	vec3  scatterDistance;
	float flatness;
	float diffTrans;
	bool  thin;
};

void Init(out DisneyMaterial mat, Material data)
{
	mat.color = data.base_color.rgb;
}

vec3 Distribution(DisneyMaterial mat, vec3 wo, vec3 wi)
{
	vec3 distribution = vec3(0.0);

	// Initialize
	vec3  c              = mat.color;
	float metallicWeight = mat.metallic;
	float e              = mat.refraction;
	float strans         = mat.specTrans;
	float diffuseWeight  = (1.0 - metallicWeight) * (1.0 - strans);
	float dt             = mat.diffTrans / 2.0;
	float rough          = mat.roughness;
	float lum            = dot(vec3(0.212671, 0.715160, 0.072169), c);
	vec3  Ctint          = lum > 0.0 ? (c / lum) : vec3(1.0);

	float sheenWeight = mat.sheen;
	vec3  Csheen;
	if (sheenWeight > 0.0)
	{
		float stint = mat.sheenTint;
		Csheen      = mix(vec3(1.0), Ctint, stint);
	}

	if (diffuseWeight > 0.0)
	{
		if (mat.thin)
		{
			float flat_ = mat.flatness;

			// TODO: Add Disney Diffuse
			// TODO: Add Disney Fake SS
		}
		else
		{
			vec3 sd = mat.scatterDistance;

			if (sd == vec3(0.0))
			{
				// TODO: Add Disney Diffuse
			}
			else
			{
				// TODO: Add Specular Transmission
				// TODO: Add Disney BSSRDF
			}
		}

		// TODO: Add Disney Retro

		if (sheenWeight > 0)
		{
			// TODO: Add Disney Sheen
		}
	}

	float aspect = sqrt(1.0 - mat.anisotropic * 0.9);
	float ax     = max(0.001, rough / aspect);
	float ay     = max(0.001, rough * aspect);

	// TODO: Define Disney Microfacet Distribution

	float specTint = mat.specularTint;
	vec3  Cspec0   = mix(SchlickR0FromEta(e) * mix(vec3(1.0), Ctint, specTint), c, metallicWeight);

	// TODO: Define Disney Fresnel
	// TODO: Add Microfacet Reflection

	float cc = mat.clearcoat;
	if (cc > 0.0)
	{
		// TODO: Add Disney Clearcoat
	}

	if (strans > 0.0)
	{
		vec3 T = strans * sqrt(c);
		if (mat.thin)
		{
			float rscaled = (0.65 * e - 0.35) * rough;
			float ax      = max(0.001, (rscaled * rscaled) / aspect);
			float ay      = max(0.001, (rscaled * rscaled) * aspect);

			TrowbridgeReitzDistribution dist;
			Init(dist, ax, ay, true);

			// TODO: Add Microfacet Transmission
		}
		else
		{
			// TODO: Add Microfacet Transmission
		}
	}

	if (mat.thin)
	{
		// Add Lambertian Transmission
	}

	return vec3(0.0);
}

vec3 SampleDistribution(in DisneyMaterial mat, in vec3 wo, inout uint seed, out vec3 wi, out float pdf)
{
	return vec3(0.0);
}

////////////// Material Sampling//////////////
vec3 Distribution(Material mat, vec3 wo, vec3 wi)
{
	if (mat.material_type == Material_Matte)
	{
		MatteMaterial matte;
		Init(matte, mat);
		return Distribution(matte, wo, wi);
	}
	else if (mat.material_type == Material_Plastic)
	{
		PlasticMaterial plastic;
		Init(plastic, mat);
		return Distribution(plastic, wo, wi);
	}
	else if (mat.material_type == Material_Metal)
	{
		MetalMaterial metal;
		Init(metal, mat);
		return Distribution(metal, wo, wi);
	}
	else if (mat.material_type == Material_Mirror)
	{
		MirrorMaterial mirror;
		Init(mirror, mat);
		return Distribution(mirror, wo, wi);
	}
	else if (mat.material_type == Material_Substrate)
	{
		SubstrateMaterial substrate;
		Init(substrate, mat);
		return Distribution(substrate, wo, wi);
	}
	else if (mat.material_type == Material_Substrate)
	{
		GlassMaterial glass;
		Init(glass, mat);
		return Distribution(glass, wo, wi);
	}

	return vec3(0.0);
}

vec3 SampleDistribution(in Material mat, in vec3 wo, inout uint seed, out vec3 wi, out float pdf)
{
	if (mat.material_type == Material_Matte)
	{
		MatteMaterial matte;
		Init(matte, mat);
		return SampleDistribution(matte, wo, seed, wi, pdf);
	}
	else if (mat.material_type == Material_Plastic)
	{
		PlasticMaterial plastic;
		Init(plastic, mat);
		return SampleDistribution(plastic, wo, seed, wi, pdf);
	}
	else if (mat.material_type == Material_Metal)
	{
		MetalMaterial metal;
		Init(metal, mat);
		return SampleDistribution(metal, wo, seed, wi, pdf);
	}
	else if (mat.material_type == Material_Mirror)
	{
		MirrorMaterial mirror;
		Init(mirror, mat);
		return SampleDistribution(mirror, wo, seed, wi, pdf);
	}
	else if (mat.material_type == Material_Substrate)
	{
		SubstrateMaterial substrate;
		Init(substrate, mat);
		return SampleDistribution(substrate, wo, seed, wi, pdf);
	}
	else if (mat.material_type == Material_Glass)
	{
		GlassMaterial glass;
		Init(glass, mat);
		return SampleDistribution(glass, wo, seed, wi, pdf);
	}

	return vec3(0.0);
}

#endif