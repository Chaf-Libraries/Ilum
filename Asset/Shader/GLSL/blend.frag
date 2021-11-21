#version 450

layout(binding = 0) uniform sampler2D Image1;
layout(binding = 1) uniform sampler2D Image2;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(texture(Image1, inUV).rgb+texture(Image2, inUV).rgb, 1.0);
}