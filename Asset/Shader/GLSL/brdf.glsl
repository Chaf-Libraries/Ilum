#include "common.glsl"

vec3 LambertianDiffuse(vec3 albedo)
{
    return albedo / PI;
}

float DistributeBlinnPhong(float NoH, float roughness)
{
    float a2 = roughness * roughness;
    return pow(NoH, 2.0 / a2 - 2.0)/(PI * a2);
}

float DistributeBeckmann(float NoH, float roughness)
{
    float a2 = roughness * roughness;
    float NoH2 = NoH * NoH;
    return exp((NoH2 - 1.0)/(a2 * NoH2))/(PI * a2 * NoH2 * NoH2 + 0.00001);
}

float DistributeGGX(float NoH, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = NoH * NoH * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NoV, float roughness)
{
    float r = (roughness + 1.0);
	float k = (r*r) / 8.0;
    return NoV /(NoV * (1.0 - k) + k);
}

float GeometrySmith(float NoL, float NoV, float roughness)
{
    float ggx1 = GeometrySchlickGGX(NoL, roughness);
    float ggx2 = GeometrySchlickGGX(NoV, roughness);

    return ggx1 * ggx2;
}

vec3 FresnelSchlick(float HoV, vec3 F0)
{
    return F0 + (1.0 - F0)*pow(1.0 - HoV, 5.0);   
}

vec3 AverageFresnel(vec3 r, vec3 g)
{
    return vec3(0.087237) + 0.0230685*g - 0.0864902*g*g + 0.0774594*g*g*g
           + 0.782654*r - 0.136432*r*r + 0.278708*r*r*r
           + 0.19744*g*r + 0.0360605*g*g*r - 0.2586*g*r*r;
}

vec3 MultiScatterBRDF(vec3 albedo, vec3 Eo, vec3 Ei, float Eavg)
{
  // copper
  vec3 edgetint = vec3(0.827, 0.792, 0.678);
  vec3 F_avg = AverageFresnel(albedo, edgetint);
  
  // TODO: To calculate fms and missing energy here
  vec3 f_add = F_avg*Eavg/(1.0-F_avg*(1.0 - Eavg));
  vec3 f_ms = (1.0-Eo)*(1.0-Ei)/(PI*(1.0-Eavg.r));

  return f_ms * f_add;
}
