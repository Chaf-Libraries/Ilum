#version 450

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive: enable

#include "common_buffer.h"

layout(location = 0) in vec4 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;
layout(location = 5) flat in uint inIndex;

layout(location = 0) out vec4 Albedo;
layout(location = 1) out vec4 Normal;
layout(location = 2) out vec4 Position;
layout(location = 3) out vec4 Metallic_Roughness_AO;
layout(location = 4) out vec4 Emissive;
layout(location = 5) out vec4 Instance_Vis;
layout(location = 6) out uint Entity_ID;

layout(push_constant) uniform PushBlock{
	layout(offset = 80)
    vec4 base_color;
    vec3 emissive_color;
    float metallic_factor;
    float roughness_factor;
    float emissive_intensity;
    
    uint albedo_map;
    uint normal_map;
    uint metallic_map;
    uint roughness_map;
    uint emissive_map;
	uint ao_map;
    uint entity_id;
} push_data;

layout (set = 0, binding = 1) uniform sampler2D textureArray[];

float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    Position = vec4(inPos.xyz, 1.0);

    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);

    // Albedo G-Buffer
    Albedo = push_data.albedo_map < 1024?
        texture(textureArray[nonuniformEXT(push_data.albedo_map)], inUV) * push_data.base_color : 
        push_data.base_color;
        
    // Metallic G-Buffer
    Metallic_Roughness_AO.r = push_data.metallic_map < 1024?
        texture(textureArray[nonuniformEXT(push_data.metallic_map)], inUV).r * push_data.metallic_factor : 
        push_data.metallic_factor;

    // Roughness G-Buffer
    Metallic_Roughness_AO.g = push_data.roughness_map < 1024?
        texture(textureArray[nonuniformEXT(push_data.roughness_map)], inUV).g * push_data.roughness_factor : 
        push_data.roughness_factor;

    Normal =  push_data.normal_map < 1024?
        vec4(TBN * normalize(texture(textureArray[nonuniformEXT(push_data.normal_map)], inUV).xyz * 2.0 - vec3(1.0)), 1.0) : 
        vec4(N, 1.0);    

    Emissive = push_data.emissive_map < 1024?
        vec4(texture(textureArray[nonuniformEXT(push_data.emissive_map)], inUV).rgb * push_data.emissive_color * push_data.emissive_intensity, 1.0) : 
        vec4(push_data.emissive_color * push_data.emissive_intensity, 1.0);

    Metallic_Roughness_AO.b = push_data.ao_map < 1024?
        texture(textureArray[nonuniformEXT(push_data.ao_map)], inUV).r : 
        0.0;
    
    Metallic_Roughness_AO.w=1.0;

    // Instance Visualization
    Instance_Vis = vec4(rand(vec2(inIndex)), rand(vec2(inIndex + 1)), rand(vec2(inIndex + 2)), 1);

    Entity_ID = push_data.entity_id;
}