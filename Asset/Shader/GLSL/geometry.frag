#version 450

#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec4 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;
layout(location = 5) flat in uint inIndex;
layout(location = 6) in vec3 inColor;

layout(location = 0) out vec4 Albedo;
layout(location = 1) out vec4 Normal;
layout(location = 2) out vec4 Position;
layout(location = 3) out vec4 Depth;
layout(location = 4) out vec4 Metallic_Roughness_AO;
layout(location = 5) out vec4 Emissive;


struct MaterialData
{
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

    float displacement_height;
    uint displacement_map;
    uint id;
};

layout (set = 0, binding = 1) uniform sampler2D textureArray[];

layout (set = 0, binding = 2) buffer MaterialBuffer
{
    MaterialData material_data[];
};

void main() {
    Position = vec4(inPos.xyz, 1.0);
    Depth = vec4(vec3(inPos.w), 1.0);

    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);

    // Albedo G-Buffer
    Albedo = material_data[inIndex].albedo_map < 1024?
        texture(textureArray[nonuniformEXT(material_data[inIndex].albedo_map)], inUV) * material_data[inIndex].base_color : 
        material_data[inIndex].base_color;

    // Albedo = vec4(inColor ,1.0);
        
    // Metallic G-Buffer
    Metallic_Roughness_AO.r = material_data[inIndex].metallic_map < 1024?
        texture(textureArray[nonuniformEXT(material_data[inIndex].metallic_map)], inUV).r * material_data[inIndex].metallic_factor : 
        material_data[inIndex].metallic_factor;

    // Roughness G-Buffer
    Metallic_Roughness_AO.g = material_data[inIndex].roughness_map < 1024?
        texture(textureArray[nonuniformEXT(material_data[inIndex].roughness_map)], inUV).g * material_data[inIndex].roughness_factor : 
        material_data[inIndex].roughness_factor;

    Normal =  material_data[inIndex].normal_map < 1024?
        vec4(TBN * normalize(texture(textureArray[nonuniformEXT(material_data[inIndex].normal_map)], inUV).xyz * 2.0 - vec3(1.0)), 1.0) : 
        vec4(N, 1.0);    

    Emissive = material_data[inIndex].emissive_map < 1024?
        vec4(texture(textureArray[nonuniformEXT(material_data[inIndex].emissive_map)], inUV).rgb * material_data[inIndex].emissive_color * material_data[inIndex].emissive_intensity, 1.0) : 
        vec4(material_data[inIndex].emissive_color * material_data[inIndex].emissive_intensity, 1.0);

    Metallic_Roughness_AO.b = material_data[inIndex].ao_map < 1024?
        texture(textureArray[nonuniformEXT(material_data[inIndex].ao_map)], inUV).r : 
        0.0;
    
    Metallic_Roughness_AO.w=1.0;
}