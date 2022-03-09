#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_shader_draw_parameters : require

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout (set = 0, binding = 0) uniform sampler2D textureArray[];

layout(push_constant) uniform PushBlock{
    layout(offset = 64)
    uint idx;
}push_data;

#define PI 3.1415926535

vec2 sampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv.x/=2*PI;
    uv.y/=PI;
    uv+=0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}

void main()
{
    vec2 uv = sampleSphericalMap(normalize(inPos));

    outColor = vec4(texture(textureArray[nonuniformEXT(push_data.idx)], uv).rgb, 1.0);
}