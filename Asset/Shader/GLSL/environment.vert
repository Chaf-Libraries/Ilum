#version 450

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_shader_draw_parameters : require

layout(location = 0) in vec3 inPos;

layout(location = 0) out vec3 outUVW;

layout (set = 0, binding = 0) uniform MainCamera
{
    mat4 view_projection;
	mat4 last_view_projection;
	vec4 frustum[6];
	vec3 position;
}main_camera;

void main()
{
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	outUVW = inPos;

    vec4 clip_pos = main_camera.view_projection * vec4(inPos + main_camera.position, 1.0);

    gl_Position  = clip_pos.xyww;
}