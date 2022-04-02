#version 450


#extension GL_ARB_shader_viewport_layer_array : enable

layout(location = 0) in vec4 inPos;
layout(location = 1) in vec3 inLightPos;
layout(location = 2) in float inDepthBias;

void main()
{
	float light_distance = (length(inPos.xyz - inLightPos.xyz) + inDepthBias) / 100.0;

	gl_FragDepth = light_distance;
}