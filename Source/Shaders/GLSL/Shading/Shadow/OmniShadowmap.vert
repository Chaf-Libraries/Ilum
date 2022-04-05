#version 450


#extension GL_ARB_shader_viewport_layer_array : enable

#include "../../GlobalBuffer.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec4 outPos;
layout(location = 1) out vec3 outLightPos;
layout(location = 2) out float outDepthBias;

layout(set = 0, binding = 0) buffer PerInstanceBuffer
{
	PerInstanceData instance_data[ ];
};

layout(set = 0, binding = 1) buffer PointLightBuffer
{
	PointLight point_lights[ ];
};

layout(push_constant) uniform PushBlock{
		mat4 transform;
		mat4 view_projection;
		vec3 light_pos;
		uint  dynamic;
		uint  light_id;
		uint  face_id;
		float depth_bias;
}push_data;

void main()
{
	mat4 trans = push_data.dynamic == 1? push_data.transform : instance_data[gl_InstanceIndex].transform;
	gl_Position =  push_data.view_projection * trans * vec4(inPos, 1.0);
	gl_Layer = int(push_data.light_id * 6 + push_data.face_id);
	
	outPos = trans * vec4(inPos, 1.0);
	outLightPos = push_data.light_pos;
	outDepthBias = push_data.depth_bias;
}