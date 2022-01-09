#version 450

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive: enable

#include "../common_buffer.glsl"

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

layout (set = 0, binding = 0) uniform MainCamera
{
    CameraData main_camera;
};

layout (set = 0, binding = 1) uniform sampler2D textureArray[];

layout(push_constant) uniform PushBlock{
	mat4 model;
    float displacement_height;
    uint displacement_map;
    uint instance_id;
} push_data;

void main()
{
    float height = push_data.displacement_map < 1024?
        max(textureLod(textureArray[nonuniformEXT(push_data.displacement_map)], inUV, 0.0).r, 0.0) * push_data.displacement_height:
        0.0;

    mat4 trans = push_data.model;

    // World normal
    mat3 mNormal = transpose(inverse(mat3(trans)));
    outNormal = mNormal * normalize(inNormal);
    outTangent = mNormal * normalize(inTangent);
    outBiTangent = mNormal * normalize(inBiTangent);

    outPos = trans * vec4(inPos, 1.0);

    outPos.xyz += push_data.displacement_map < 1024?
        normalize(outNormal) * (max(textureLod(textureArray[nonuniformEXT(push_data.displacement_map)], inUV, 0.0).r, 0.0) * push_data.displacement_height):
        vec3(0.0);

    gl_Position = main_camera.view_projection * outPos;

    outPos.w = gl_Position.z;

    outUV = inUV;

    outIndex = push_data.instance_id;
}