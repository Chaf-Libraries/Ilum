#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;

layout(location = 0) out vec3 outPos;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out vec3 outTangent;
layout(location = 4) out vec3 outBiTangent;
layout(location = 5) out vec4 outScreenPos;

layout (set = 0, binding = 0) uniform MainCamera
{
    mat4 view_projection;
}main_camera;

layout(push_constant) uniform PushConsts {
	mat4 model;
} pushConsts;

void main() {
    gl_Position = main_camera.view_projection*pushConsts.model*vec4(inPos, 1.0);
    outScreenPos = gl_Position;

    outPos = vec3(inPos);
    outUV = inUV;
    outNormal = normalize(inNormal);
    outTangent = normalize(inTangent);
    outBiTangent = normalize(inBiTangent);
}