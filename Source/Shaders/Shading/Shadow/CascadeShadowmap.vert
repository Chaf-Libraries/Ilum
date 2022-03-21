#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_ARB_shader_viewport_layer_array : enable

#include "../../GlobalBuffer.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;

layout(location = 0) out int outInstanceIndex;

void main()
{
	outInstanceIndex = gl_InstanceIndex;
	gl_Position =  vec4(inPos, 1.0);
}