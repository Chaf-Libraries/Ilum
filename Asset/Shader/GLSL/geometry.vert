#version 450

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_shader_draw_parameters : require

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
layout(location = 6) out vec3 outColor;

struct MaterialData
{
    vec4 base_color;
    vec3 emissive_color;
    float metallic_factor;

    float roughness_factor;
    float emissive_intensity;
    uint albedo_map;
    uint normal_map;

    uint metallic_map;
    uint roughness_map;
    uint emissive_map;
    uint ao_map;

    float displacement_height;
    uint displacement_map;
    uint id;
};

struct TransformData
{
    mat4 world_transform;
    mat4 pre_transform;
};

layout (set = 0, binding = 0) uniform MainCamera
{
    mat4 view_projection;
    vec4 frustum[6];
    vec3 position;
}main_camera;


layout (set = 0, binding = 1) uniform sampler2D textureArray[];

layout (set = 0, binding = 2) buffer MaterialBuffer
{
    MaterialData material_data[];
};

layout (set = 0, binding = 3) buffer TransformBuffer
{
    TransformData transform[];
};

layout (set = 0, binding = 4) buffer MeshletBuffer
{
    uint meshlets[];
};

float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    float height=material_data[meshlets[gl_DrawIDARB]].displacement_map < 1024?
        max(textureLod(textureArray[nonuniformEXT(material_data[meshlets[gl_DrawIDARB]].displacement_map)], inUV, 0.0).r, 0.0) * material_data[meshlets[gl_DrawIDARB]].displacement_height:
        0.0;
    
    mat4 trans = transform[meshlets[gl_DrawIDARB]].world_transform*transform[meshlets[gl_DrawIDARB]].pre_transform;

    // World normal
    mat3 mNormal = transpose(inverse(mat3(trans)));
    outNormal = mNormal * normalize(inNormal);
    outTangent = mNormal * normalize(inTangent);
    outBiTangent = mNormal * normalize(inBiTangent);

    outPos = trans * vec4(inPos, 1.0);

    outPos.xyz += material_data[meshlets[gl_DrawIDARB]].displacement_map < 1024?
        normalize(outNormal) * (max(textureLod(textureArray[nonuniformEXT(material_data[meshlets[gl_DrawIDARB]].displacement_map)], inUV, 0.0).r, 0.0) * material_data[meshlets[gl_DrawIDARB]].displacement_height):
        vec3(0.0);

    gl_Position = main_camera.view_projection * outPos;

    outPos.w = gl_Position.z;

    outUV = inUV;


    outIndex = meshlets[gl_DrawIDARB];

    outColor = vec3(rand(vec2(gl_DrawIDARB,gl_DrawIDARB+1)), 
    rand(vec2(gl_DrawIDARB+2,gl_DrawIDARB+3)), 
    rand(vec2(gl_DrawIDARB+4,gl_DrawIDARB+5)));
}