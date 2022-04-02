#ifndef _SAMPLING_GLSL
#define _SAMPLING_GLSL

#include "Geometry.glsl"
#include "GlobalBuffer.glsl"
#include "Random.glsl"
#include "Math.glsl"

// Sampling Disk
// Polar Mapping
vec2 UniformSampleDisk(vec2 u)
{
	float r     = sqrt(u.x);
	float theta = 2 * PI * u.y;
	return r * vec2(cos(theta), sin(theta));
}

// Concentric Mapping
vec2 SampleConcentricDisk(vec2 u)
{
	vec2 uOffset = 2.0 * u - 1.0;

	if (uOffset.x == 0 && uOffset.y == 0)
	{
		return vec2(0.0);
	}

	float theta, r;
	if (abs(uOffset.x) > abs(uOffset.y))
	{
		r     = uOffset.x;
		theta = PIOver4 * (uOffset.y / uOffset.x);
	}
	else
	{
		r     = uOffset.y;
		theta = PIOver2 - PIOver4 * (uOffset.x / uOffset.y);
	}

	return r * vec2(cos(theta), sin(theta));
}

// Sampling Hemisphere
vec3 UniformSampleHemisphere(vec2 u)
{
	float z   = u.x;
	float r   = sqrt(max(0.0, 1.0 - z * z));
	float phi = 2 * PI * u.y;
	return vec3(r * cos(phi), r * sin(phi), z);
}

float UniformHemispherePdf()
{
	return Inv2PI;
}

vec3 SampleCosineHemisphere(vec2 u)
{
	vec2  d = SampleConcentricDisk(u);
	float z = sqrt(max(0, 1 - d.x * d.x - d.y * d.y));
	return vec3(d.x, d.y, z);
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

// Sampling Point Light
vec3 Sample_Li(PointLight light, Interaction interaction, vec2 u, out vec3 wi, out float pdf)
{
	wi  = normalize(light.position - interaction.position);
	pdf = 1.0;

	float d    = length(light.position - interaction.position);
	float Fatt = 1.0 / (light.constant + light.linear_ * d + light.quadratic * d * d);
	return light.color.rgb * light.intensity * Fatt;
}

float Pdf_Li(PointLight light, Interaction interaction, vec3 wi)
{
	return 0.0;
}

float Power(PointLight light)
{
	return 4 * PI * light.intensity;
}

// Sampling Spot Light
vec3 Sample_Li(SpotLight light, Interaction interaction, vec2 u, out vec3 wi, out float pdf)
{
	wi  = normalize(light.position - interaction.position);
	pdf = 1.0;

	vec3  L         = normalize(light.position - interaction.position);
	float NoL       = max(0.0, dot(interaction.normal, L));
	float theta     = dot(L, normalize(-light.direction));
	float epsilon   = light.cut_off - light.outer_cut_off;
	return light.color * light.intensity * clamp((theta - light.outer_cut_off) / epsilon, 0.0, 1.0);
}

float Pdf_Li(SpotLight light, Interaction interaction, vec3 wi)
{
	return 0.0;
}

float Power(SpotLight light)
{
	return light.intensity * 2.0 * PI * (1.0 - 0.5 * (light.cut_off - light.outer_cut_off));
}

// Sampling Directional Light
vec3 Sample_Li(DirectionalLight light, Interaction interaction, vec2 u, out vec3 wi, out float pdf)
{
	wi  = normalize(-light.direction);
	pdf = 1.0;

	return light.color.rgb * light.intensity;
}

float Pdf_Li(DirectionalLight light, Interaction interaction, vec3 wi)
{
	return 0.0;
}

float Power(DirectionalLight light)
{
	return Infinity;
}


#endif