#version 450

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_shader_draw_parameters : require

layout(location = 0) out vec3 outPos;
layout(location = 1) out vec2 outUV;

layout(push_constant) uniform PushBlock{
    mat4 view_projection;
}push_data;

void main()
{
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	gl_Position = vec4(uv * 2.0f - 1.0f, 1.0f, 1.0f);

    outUV=uv;

    outPos = (inverse(push_data.view_projection) * gl_Position).xyz;
}

