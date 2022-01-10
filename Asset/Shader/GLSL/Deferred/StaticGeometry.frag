#version 450

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive: enable

#include "../common_buffer.glsl"
#include "../common.glsl"

layout(location = 0) in vec4 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;
layout(location = 5) flat in uint inIndex;
layout(location = 6) in vec4 inScreenSpacePos;
layout(location = 7) in vec4 inlastScreenSpacePos;

layout(location = 0) out vec4 Albedo;
layout(location = 1) out vec4 Normal;
layout(location = 2) out vec4 Position;
layout(location = 3) out vec4 Metallic_Roughness_AO;
layout(location = 4) out vec4 Emissive;
layout(location = 5) out vec4 MotionVector_Curvature;
layout(location = 6) out float LinearDepth;
layout(location = 7) out uint Entity_ID;

layout (set = 0, binding = 1) uniform sampler2D textureArray[];

layout (set = 0, binding = 2) buffer PerInstanceBuffer
{
    PerInstanceData instance_data[];
};

layout (set = 0, binding = 3) buffer MaterialBuffer
{
    MaterialData material_data[];
};

float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    Position = vec4(inPos.xyz, 1.0);

    MotionVector_Curvature = vec4(compute_motion_vector(inlastScreenSpacePos, inScreenSpacePos), 0.0, 1.0);

    LinearDepth = inPos.w;

    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);

    // Albedo G-Buffer
    Albedo = material_data[inIndex].albedo_map < 1024?
        texture(textureArray[nonuniformEXT(material_data[inIndex].albedo_map)], inUV) * material_data[inIndex].base_color : 
        material_data[inIndex].base_color;
        
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

    Entity_ID = instance_data[inIndex].entity_id;
}