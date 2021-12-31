#version 450

#extension GL_GOOGLE_include_directive: enable

#include "light.h"

#include "brdf.glsl"

layout(binding = 0) uniform sampler2D Albedo;
layout(binding = 1) uniform sampler2D Normal;
layout(binding = 2) uniform sampler2D Position_Depth;
layout(binding = 3) uniform sampler2D Metallic_Roughness_AO;
layout(binding = 4) uniform sampler2D Emissive;
layout(binding = 5) uniform sampler2D Emu_Lut;
layout(binding = 6) uniform sampler2D Eavg_Lut;

layout(binding = 7) buffer DirectionalLights{
    DirectionalLight directional_lights[ ];
};

layout(binding = 8) buffer PointLights{
    PointLight point_lights[ ];
};

layout(binding = 9) buffer SpotLights{
    SpotLight spot_lights[ ];
};

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outColor;

layout(push_constant) uniform PushBlock{
    vec3 view_pos;
    uint directional_light_count;
    uint spot_light_count;
    uint point_light_count;
}push_data;

// From http://filmicgames.com/archives/75
vec3 uncharted2Tonemap(vec3 x)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

vec3 specularContribution(vec3 L, vec3 V, vec3 N, vec3 F0, float metallic, float roughness, vec3 albedo, vec3 radiance)
{
    vec3 H = normalize(V + L);
    float NoV = clamp(dot(N,V), 0.0, 1.0);
    float NoL = clamp(dot(N,L), 0.0, 1.0);
    float NoH = clamp(dot(N,H), 0.0, 1.0);
    float HoV = clamp(dot(H,V), 0.0, 1.0);

    vec3 Eo = vec3(texture(Emu_Lut, vec2(NoL, roughness)).r);
    vec3 Ei = vec3(texture(Emu_Lut, vec2(NoV, roughness)).r);
    float Eavg = texture(Eavg_Lut, vec2(0, roughness)).r;

    float D = DistributeGGX(NoH, roughness);
    float G = GeometrySmith(NoL, NoV, roughness);
    vec3 F = FresnelSchlick(HoV, F0);

    vec3 specular = D * F * G / (4.0 * NoL * NoV + 0.001);
    vec3 Kd = (vec3(1.0)-F)*(1-metallic);

    vec3 Fmicro = Kd * albedo/PI+specular;

    vec3 Fms = MultiScatterBRDF(albedo, Eo, Ei, Eavg);
    vec3 BRDF = Fmicro + Fms;

    return BRDF * radiance * NoL;
}

void main()
{
    vec3 albedo = pow(texture(Albedo, inUV).rgb, vec3(2.2));
    vec3 normal = texture(Normal, inUV).rgb;
    vec3 emissive = texture(Emissive, inUV).rgb;
    vec3 frag_pos = texture(Position_Depth, inUV).rgb;
    float metallic = texture(Metallic_Roughness_AO, inUV).r;
    float roughness = texture(Metallic_Roughness_AO, inUV).g;

    roughness = roughness == 0.0 ? 6.274e-5 : roughness;

    vec3 Lo = vec3(0.0);
    outColor = vec4(0.0,0.0,0.0,1.0);
    vec3 V = normalize(push_data.view_pos - frag_pos);
    vec3 N = normalize(normal);

    vec3 F0 = vec3(0.0);
    F0 = mix(F0, albedo, metallic);

    // Directional light
    for(uint i = 0; i < push_data.directional_light_count; i++)
    {
        vec3 L = normalize(directional_lights[i].direction);
        vec3 radiance = directional_lights[i].color.rgb * directional_lights[i].intensity;
        Lo += specularContribution(L, V, N, F0, metallic, roughness, albedo.rgb, radiance);
    }

    for(uint i = 0; i< push_data.spot_light_count; i++)
    {
        vec3 L = normalize(spot_lights[i].position - frag_pos);
        float NoL = max(0.0, dot(N,L));
        float theta = dot(L, normalize(-spot_lights[i].direction));
        float epsilon   = spot_lights[i].cut_off - spot_lights[i].outer_cut_off;
        float intensity = spot_lights[i].intensity * clamp((theta - spot_lights[i].outer_cut_off) / epsilon, 0.0, 1.0);
        vec3 radiance = spot_lights[i].color * intensity;
        Lo += specularContribution(L, V, N, F0, metallic, roughness, albedo.rgb, radiance);
    }

    for(uint i = 0; i< push_data.point_light_count; i++)
    {
        vec3 L = normalize(point_lights[i].position - frag_pos);
        float d = length(point_lights[i].position - frag_pos);
        float NoL = max(0.0, dot(N,L));
        float Fatt = 1.0/(point_lights[i].constant + point_lights[i].linear * d + point_lights[i].quadratic * d * d);
        vec3 radiance = point_lights[i].color.rgb * point_lights[i].intensity * Fatt;
        Lo += specularContribution(L, V, N, F0, metallic, roughness, albedo.rgb, radiance);
    }

    vec3 color = Lo + emissive;

    outColor = vec4(color, 1.0);
}