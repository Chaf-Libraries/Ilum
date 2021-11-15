#version 450

#extension GL_EXT_nonuniform_qualifier : require

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

layout (set = 0, binding = 1) uniform sampler2D textureArray[];

layout (set = 0, binding = 0) uniform MainCamera
{
    mat4 view_projection;
}main_camera;

layout(push_constant) uniform PushBlock{
	mat4 model;
    float displacement_height;
    uint displacement_map;
} model;



void main() {
    float height=model.displacement_map < 1024?
        max(textureLod(textureArray[nonuniformEXT(model.displacement_map)], inUV, 0.0).r, 0.0) * model.displacement_height:
        0.0;
    vec4 pos = model.model*vec4(inPos+height*inNormal, 1.0);

    gl_Position = main_camera.view_projection* (pos);

    // World position
    outPos.xyz = (pos / pos.w).xyz;
    // Depth
    outPos.w = gl_Position.z;

    outUV = inUV;

    // World normal
    mat3 mNormal = transpose(inverse(mat3(model.model)));
    outNormal = mNormal * normalize(inNormal);
    outTangent = mNormal * normalize(inTangent);
    outBiTangent = mNormal * normalize(inBiTangent);

    // displacement
    gl_Position.xyz += model.displacement_map < 1024?
        normalize(inNormal) * (max(textureLod(textureArray[nonuniformEXT(model.displacement_map)], inUV, 0.0).r, 0.0) * model.displacement_height):
        vec3(0.0);
}