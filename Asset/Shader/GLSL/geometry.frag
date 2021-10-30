#version 450

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_shader_draw_parameters: enable

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;
layout(location = 5) in vec4 inScreenPos;

layout(location = 0) out vec4 Normal;
layout(location = 1) out vec4 Position_Depth;

layout (set = 0, binding = 1) uniform sampler2D textureArray[];

void main() {
    Position_Depth.xyz = inPos;
    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    Normal =  vec4(inNormal, 1.0);
    Position_Depth.w=1;
    // Position_Depth.w = 1.0;
    // Position_Depth=vec4(1.0);
   // Normal=vec4(1.0);
}