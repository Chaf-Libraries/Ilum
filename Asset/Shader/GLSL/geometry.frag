#version 450

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_shader_draw_parameters: enable

layout(location = 0) in vec4 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;

layout(location = 0) out vec4 Albedo;
layout(location = 1) out vec4 Normal;
layout(location = 2) out vec4 Position;
layout(location = 3) out vec4 Depth;
layout(location = 4) out vec4 Metallic;
layout(location = 5) out vec4 Roughness;
layout(location = 6) out vec4 Emissive;
layout(location = 7) out vec4 AO;

layout (set = 0, binding = 1) uniform sampler2D textureArray[];

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
    float entity_id;
}material;


void main() {
    Position = vec4(inPos.xyz, 1.0);
    Depth = vec4(vec3(inPos.w), 1.0);

    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);

    // Albedo G-Buffer
    Albedo = material.albedo_map < 1024?
        texture(textureArray[nonuniformEXT(material.albedo_map)], inUV) * material.base_color : 
        material.base_color;
    
    // Metallic G-Buffer
    Metallic = material.metallic_map < 1024?
        vec4(vec3(texture(textureArray[nonuniformEXT(material.metallic_map)], inUV).r * material.metallic_factor), 1.0) : 
        vec4(vec3(material.metallic_factor), 1.0);

    // Roughness G-Buffer
    Roughness = material.roughness_map < 1024?
        vec4(vec3(texture(textureArray[nonuniformEXT(material.roughness_map)], inUV).g * material.roughness_factor), 1.0) : 
        vec4(vec3(material.roughness_factor), 1.0);

    Normal =  material.normal_map < 1024?
        vec4(TBN * normalize(texture(textureArray[nonuniformEXT(material.normal_map)], inUV).xyz * 2.0 - vec3(1.0)), 1.0) : 
        vec4(N, 1.0);    

    Emissive = material.emissive_map < 1024?
        texture(textureArray[nonuniformEXT(material.emissive_map)], inUV) * vec4(material.emissive_color * material.emissive_intensity, 1.0) : 
        vec4(material.emissive_color * material.emissive_intensity, 1.0);

    AO = material.ao_map < 1024?
        vec4(vec3(texture(textureArray[nonuniformEXT(material.ao_map)], inUV).r), 1.0) : 
        vec4(vec3(0.0), 1.0);     
}