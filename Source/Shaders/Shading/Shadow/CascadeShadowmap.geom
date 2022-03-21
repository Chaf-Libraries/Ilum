#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_ARB_shader_viewport_layer_array : enable

#include "../../GlobalBuffer.glsl"

layout (triangles, invocations = 4) in;
layout (triangle_strip, max_vertices = 3) out;

layout (location = 0) in int inInstanceIndex[];

layout(set = 0, binding = 0) buffer PerInstanceBuffer
{
	PerInstanceData instance_data[ ];
};

layout(set = 0, binding = 1) buffer DirectionalLights{
    DirectionalLight directional_lights[ ];
};

layout(push_constant) uniform PushBlock{
	mat4 transform;
	uint dynamic;
	uint light_id;
}push_data;

void main() 
{
	mat4 trans = push_data.dynamic == 1? push_data.transform : instance_data[inInstanceIndex[0]].transform;
	
	for (int i = 0; i < gl_in.length(); i++)
	{
		gl_Position =  directional_lights[push_data.light_id].view_projection[gl_InvocationID] * trans * gl_in[i].gl_Position;
		gl_Layer = int(push_data.light_id * 4 + gl_InvocationID);
		EmitVertex();
	}
	EndPrimitive();
}
