#version 450


#extension GL_ARB_shader_viewport_layer_array : enable

#include "../../GlobalBuffer.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;

layout(set = 0, binding = 0) buffer PerInstanceBuffer
{
	PerInstanceData instance_data[ ];
};

layout(set = 0, binding = 1) buffer SpotLightBuffer
{
	SpotLight spot_lights[ ];
};

layout(push_constant) uniform PushBlock{
	mat4 transform;
	uint dynamic;
	uint layer;
}push_data;

void main()
{
	mat4 trans = push_data.dynamic == 1? push_data.transform : instance_data[gl_InstanceIndex].transform;
	gl_Position =  spot_lights[push_data.layer].view_projection * trans * vec4(inPos, 1.0);
	gl_Layer = int(push_data.layer);
}