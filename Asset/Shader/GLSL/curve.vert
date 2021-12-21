#version 450

layout(location = 0) in vec3 inPos;

layout(location = 0) out vec4 outColor;
layout(location = 1) out uint outInstanceID;
layout(location = 2) out uint outEntityID;

layout (set = 0, binding = 0) uniform MainCamera
{
    mat4 view_projection;
	mat4 last_view_projection;
	vec4 frustum[6];
	vec3 position;
}main_camera;

layout(push_constant) uniform PushBlock{
	mat4 transform;
    vec4 color;
    uint entity_id;
    uint instance_id;
} push_data;

void main() 
{
    gl_Position = main_camera.view_projection * push_data.transform * vec4(inPos, 1.0);
    outColor = push_data.color;
    outEntityID = push_data.entity_id;
    outInstanceID = push_data.instance_id;
}