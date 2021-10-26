#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;
layout(location = 5) in vec4 inScreenPos;

layout(location = 0) out vec4 Normal;
layout(location = 1) out vec4 Position_Depth;

void main() {
    Position_Depth.xyz = inPos;
    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    Normal.xyz = inNormal;
    Normal.w = 1;
    Position_Depth.w=1;
    // Position_Depth.w = 1.0;
    // Position_Depth=vec4(1.0);
   // Normal=vec4(1.0);
}