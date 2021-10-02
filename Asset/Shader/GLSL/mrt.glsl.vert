#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;

layout(location = 0) out vec3 outPos;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out vec3 outTangent;
layout(location = 4) out vec3 outBiTangent;
layout(location = 5) out vec4 outScreenPos;

layout(set = 0, binding = 0) uniform UBO
{
    mat4 viewProjection;
}ubo;

struct UniformBufferFrame
{
    mat4 view_projection;
};

struct UniformBufferInstance
{
    mat4 transform;
    mat4 prev_transform;
    
    vec4 albedo_factor;
    
    float roughness_factor;
    float metallic_factor;
    float normal_factor;
    float displacement_factor;
};

layout(set = 0, binding = 1) uniform FrameUBO
{
    UniformBufferFrame data;
}frame_ubo;

layout(set = 0, binding = 2) uniform InstanceUBO
{
    UniformBufferInstance data;
}instance_ubo;

layout(push_constant) uniform PushConsts {
	mat4 model;
} push_constants;

void main() {
    // flip Y
    gl_Position = frame_ubo.data.view_projection*instance_ubo.data.transform*vec4(inPos, 1.0);
    outScreenPos = gl_Position;

    outPos = vec3(push_constants.model*vec4(inPos, 1.0));
    outUV = inUV;
    mat3 mNormal = transpose(inverse(mat3(push_constants.model)));
    outNormal = mNormal*normalize(inNormal);
    outTangent = mNormal*normalize(inTangent);
    outBiTangent = mNormal*normalize(inBiTangent);
}