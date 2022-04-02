#version 450


#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_shader_draw_parameters : enable
#extension GL_GOOGLE_include_directive : enable

#include "../../GlobalBuffer.glsl"

layout(location = 0) in vec4 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;
layout(location = 5) flat in uint inIndex;
layout(location = 6) in vec4 inScreenSpacePos;
layout(location = 7) in vec4 inLastScreenSpacePos;
layout(location = 8) flat in uint inEntityID;

layout(location = 0) out vec4 GBuffer0; // RGB - Albedo, A - Anisotropic
layout(location = 1) out vec4 GBuffer1;   // RGB - Normal, A - Linear Depth
layout(location = 2) out vec4 GBuffer2; // R - Metallic, G - Roughness, B - Subsurface, A - EntityID
layout(location = 3) out vec4 GBuffer3; // R - Sheen, G - Sheen Tint, B - Clearcoat, A - Clearcoat Gloss
layout(location = 4) out vec4 GBuffer4; // RG - Velocity, B - Specular, A - Specular Tint
layout(location = 5) out vec4 GBuffer5; // RGB - Emissive, A - Material Type

layout (set = 0, binding = 1) uniform sampler2D texture_array[];

layout (set = 0, binding = 2) buffer PerInstanceBuffer
{
    PerInstanceData instance_data[ ];
};

layout (set = 0, binding = 4) buffer MaterialBuffer
{
    MaterialData material_data[ ];
};

float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

vec2 compute_motion_vector(vec4 prev_pos, vec4 current_pos)
{
    // Clip space -> NDC
    vec2 current = current_pos.xy / current_pos.w;
    vec2 prev = prev_pos.xy / prev_pos.w;

    current = current * 0.5 + 0.5;
    prev = prev * 0.5 + 0.5;

    current.y = 1 - current.y;
    prev.y = 1 - prev.y;

    return current - prev;
}


void main() {
    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);

    // GBuffer0
    // RGB - Albedo
    GBuffer0.rgb = material_data[inIndex].textures[TEXTURE_BASE_COLOR] < MAX_TEXTURE_ARRAY_SIZE?
        texture(texture_array[nonuniformEXT(material_data[inIndex].textures[TEXTURE_BASE_COLOR])], inUV) .rgb
        * material_data[inIndex].base_color.rgb : material_data[inIndex].base_color.rgb;
    // A - Anisotropic
    GBuffer0.a = material_data[inIndex].anisotropic;

    // GBuffer1
    // RGB - Normal
    GBuffer1.rgb =  material_data[inIndex].textures[TEXTURE_NORMAL] < MAX_TEXTURE_ARRAY_SIZE?
        TBN * normalize(texture(texture_array[nonuniformEXT(material_data[inIndex].textures[TEXTURE_NORMAL])], inUV).xyz * 2.0 - vec3(1.0)) : 
        N;
     // A - Linear Depth
     GBuffer1.a = inPos.w;

     // GBuffer2
     // R - Metallic
     GBuffer2.r = material_data[inIndex].textures[TEXTURE_METALLIC] < 1024?
         texture(texture_array[nonuniformEXT(material_data[inIndex].textures[TEXTURE_METALLIC])], inUV).r * material_data[inIndex].metallic : 
         material_data[inIndex].metallic;
     // G - Roughness
     GBuffer2.g = material_data[inIndex].textures[TEXTURE_ROUGHNESS] < 1024?
         texture(texture_array[nonuniformEXT(material_data[inIndex].textures[TEXTURE_ROUGHNESS])], inUV).g * material_data[inIndex].roughness : 
         material_data[inIndex].roughness;
     // B - Subsurface
     GBuffer2.b = material_data[inIndex].subsurface;
     // A - EntityID
     GBuffer2.a = inEntityID;
 
     // GBuffer3
     // R - Sheen
     GBuffer3.r = material_data[inIndex].sheen;
     // G - Sheen Tint
     GBuffer3.g = material_data[inIndex].sheen_tint;
     // B - Clearcoat
     GBuffer3.b = material_data[inIndex].clearcoat;
     // A - Clearcoat Gloss
     GBuffer3.a = material_data[inIndex].clearcoat_gloss;
 
     // GBuffer4
     // RG - Velocity
     GBuffer4.rg = compute_motion_vector(inLastScreenSpacePos, inScreenSpacePos);
     // B - Specular
     GBuffer4.b = material_data[inIndex].specular;
     // A - Specular Tint
     GBuffer4.a = material_data[inIndex].specular_tint;

     // GBuffer5
    // RGB - Emissive
    GBuffer5.rgb =  material_data[inIndex].textures[TEXTURE_EMISSIVE] < MAX_TEXTURE_ARRAY_SIZE?
        texture(texture_array[nonuniformEXT(material_data[inIndex].textures[TEXTURE_EMISSIVE])], inUV).xyz * material_data[inIndex].emissive_intensity : 
        material_data[inIndex].emissive_color * material_data[inIndex].emissive_intensity;
    GBuffer5.a = material_data[inIndex].material_type;
}