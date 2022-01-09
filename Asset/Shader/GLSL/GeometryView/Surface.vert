#version 450

#extension GL_GOOGLE_include_directive: enable

#include "../common_buffer.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;

layout(location = 0) out vec4 outColor;
layout(location = 1) out uint outEntityID;

layout (set = 0, binding = 0) uniform MainCamera
{
    CameraData main_camera;
};

layout(push_constant) uniform PushBlock{
	mat4 transform;
    uint entity_id;
    uint instance_id;
} push_data;

void main() 
{
    gl_Position = main_camera.view_projection * push_data.transform * vec4(inPos, 1.0);
    outColor = vec4(inUV, 0.0, 1.0);
    outEntityID = push_data.entity_id;
}