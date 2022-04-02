#version 450


#extension GL_GOOGLE_include_directive: enable

#include "../GlobalBuffer.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;

layout(location = 0) out vec2 outUV;
layout(location = 1) out uint outTextureID;

layout (set = 0, binding = 0) uniform MainCamera
{
    CameraData main_camera;
};

layout(push_constant) uniform PushBlock{
	mat4 transform;
    uint texture_id;
    uint parameterization;
} push_data;

void main() 
{
    if(push_data.parameterization == 1)
    {
        gl_Position = vec4(2.0 * inUV - 1.0, 0.0, 1.0);
    }
    else
    {
        gl_Position = main_camera.view_projection * push_data.transform * vec4(inPos, 1.0);
    }
    
    outUV = inUV;
    outTextureID = push_data.texture_id;
}