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

layout (set = 0, binding = 0) uniform MainCamera
{
    mat4 view_projection;
    vec3 position;
}main_camera;

layout(push_constant) uniform Model{
	mat4 model;
} model;



void main() {
    gl_Position = main_camera.view_projection*model.model*vec4(inPos, 1.0);
    outPos = vec3(model.model*vec4(inPos, 1.0));
    outUV = inUV;
    outNormal = mat3(transpose(inverse(model.model)))*inNormal;
    outTangent = mat3(transpose(inverse(model.model)))*inTangent;
    outBiTangent = mat3(transpose(inverse(model.model)))*inBiTangent;
}