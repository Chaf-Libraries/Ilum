#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_shader_draw_parameters : require
#extension GL_GOOGLE_include_directive : enable

layout(location = 0) in vec2 inUV;
layout(location = 1) in flat uint inTextureID;

layout(location = 0) out vec4 Output;

layout (set = 0, binding = 1) uniform sampler2D texture_array[];

void main()
{
    if(inTextureID < 1024)
    {
        Output = vec4(texture(texture_array[nonuniformEXT(inTextureID)], inUV).rgb, 1.0);
    }
    else
    {
        Output = vec4(inUV, 0.0, 1.0);
    }
}