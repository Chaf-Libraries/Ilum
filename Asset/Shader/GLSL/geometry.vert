#version 450

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_shader_draw_parameters : require

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;

layout(location = 0) out vec4 outPos;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out vec3 outTangent;
layout(location = 4) out vec3 outBiTangent;
layout(location = 5) out uint outIndex;

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

layout (set = 0, binding = 0) uniform MainCamera
{
    mat4 view_projection;
    vec4 frustum[6];
    vec3 position;
}main_camera;


layout (set = 0, binding = 1) uniform sampler2D textureArray[];

layout (set = 0, binding = 2) buffer InstanceBuffer
{
    MaterialData instance_data[];
};

layout (set = 0, binding = 3) buffer TransformBuffer
{
    mat4 transform[];
};


void main() {
    float height=instance_data[gl_DrawIDARB].displacement_map < 1024?
        max(textureLod(textureArray[nonuniformEXT(instance_data[gl_DrawIDARB].displacement_map)], inUV, 0.0).r, 0.0) * instance_data[gl_DrawIDARB].displacement_height:
        0.0;
    outPos = transform[gl_DrawIDARB]*vec4(inPos+height*inNormal, 1.0);

    gl_Position = main_camera.view_projection * outPos;

    outUV = inUV;

    // World normal
    mat3 mNormal = transpose(inverse(mat3(transform[gl_DrawIDARB])));
    outNormal = mNormal * normalize(inNormal);
    outTangent = mNormal * normalize(inTangent);
    outBiTangent = mNormal * normalize(inBiTangent);

    // displacement
    gl_Position.xyz += instance_data[gl_DrawIDARB].displacement_map < 1024?
        normalize(inNormal) * (max(textureLod(textureArray[nonuniformEXT(instance_data[gl_DrawIDARB].displacement_map)], inUV, 0.0).r, 0.0) * instance_data[gl_DrawIDARB].displacement_height):
        vec3(0.0);

    outIndex = gl_DrawIDARB;
}