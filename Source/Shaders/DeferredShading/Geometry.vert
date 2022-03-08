#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_shader_draw_parameters : require
#extension GL_GOOGLE_include_directive : enable

#include "../GlobalBuffer.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;

layout(location = 0) out vec4 outPos;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out vec3 outTangent;
layout(location = 4) out vec3 outBiTangent;
layout(location = 5) out uint outIndex;
layout(location = 6) out vec4 outScreenSpacePos;
layout(location = 7) out vec4 outLastScreenSpacePos;

layout(set = 0, binding = 0) uniform CameraBuffer
{
	CameraData camera_data;
};

layout (set = 0, binding = 1) uniform sampler2D TextureArray[];

layout(set = 0, binding = 2) buffer PerInstanceBuffer
{
	PerInstanceData instance_data[ ];
};

layout(set = 0, binding = 3) buffer PerMeshletBuffer
{
	PerMeshletData meshlet_data[ ];
};

layout (set = 0, binding = 4) buffer MaterialBuffer
{
    MaterialData material_data[ ];
};

layout(push_constant) uniform PushBlock
{
    mat4 transform;
    uint dynamic;
}push_data;

void main() {
    outIndex = gl_InstanceIndex;

    float height = material_data[outIndex].textures[TEXTURE_DISPLACEMENT] < MAX_TEXTURE_ARRAY_SIZE?
        max(texture(TextureArray[nonuniformEXT(material_data[outIndex].textures[TEXTURE_DISPLACEMENT])], inUV).r, 0.0) 
        * material_data[outIndex].displacement: 0.0;
    
    mat4 trans = push_data.dynamic == 1? push_data.transform: instance_data[outIndex].transform;

    // World normal
    mat3 mNormal = transpose(inverse(mat3(trans)));
    outNormal = mNormal * normalize(inNormal);
    outTangent = mNormal * normalize(inTangent);
    outBiTangent = mNormal * normalize(inBiTangent);

    outPos = trans * vec4(inPos, 1.0);

    outPos.xyz += normalize(outNormal) * height;

    gl_Position = camera_data.view_projection * outPos;

    outScreenSpacePos = gl_Position;

    outLastScreenSpacePos = push_data.dynamic == 1? camera_data.last_view_projection * instance_data[outIndex].last_transform * vec4(inPos, 1.0) : vec4(0.0);

    outPos.w = gl_Position.z;

    outUV = inUV;
}