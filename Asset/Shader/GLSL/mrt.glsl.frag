#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;
layout(location = 5) in vec4 inScreenPos;

layout(location = 0) out vec4 Albedo;
layout(location = 1) out vec4 Normal;
layout(location = 2) out vec4 Position_Depth;
layout(location = 3) out vec4 Material;

layout(set = 0, binding = 0) uniform UBO
{
    mat4 mvp;
}ubo;

layout(set = 1, binding = 0) uniform sampler bilinear;
layout(set = 1, binding = 1) uniform texture2D diffuseMap;
layout(set = 1, binding = 2) uniform texture2D normalMap;
layout(set = 1, binding = 3) uniform texture2D specularMap;

void main() {
    Position_Depth.xyz = inPos;
    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    Normal.xyz = TBN * normalize(texture(sampler2D(normalMap, bilinear), inUV).xyz*2.0 - vec3(1.0));
    Albedo.xyz = texture(sampler2D(diffuseMap, bilinear), inUV).xyz;
    Material.x = texture(sampler2D(specularMap, bilinear), inUV).x;
    Position_Depth.w = inScreenPos.z/inScreenPos.w;
}