#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shader_draw_parameters: enable

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inColor;
layout (location = 4) in vec4 inTangent;

layout (location = 5) in uint inIndex;

layout (set = 0, binding = 0) uniform UBOScene 
{
	mat4 projection;
	mat4 view;
	vec4 lightPos;
	vec4 viewPos;
    vec4 frustum[6];
	vec4 range;
} uboScene;

struct ObjectData 
{
	int baseColorTextureIndex;
	int normalTextureIndex;
	int emissiveTextureIndex;
	int occlusionTextureIndex;
	int metallicRoughnessTextureIndex;
	float metallicFactor;
	float roughnessFactor;
	int alphaMode;
	float alphaCutOff;
	uint doubleSided;

	// Parameter
	vec4 baseColorFactor;
	vec3 emissiveFactor;

	mat4 model;
};

layout (set = 0, binding = 1) buffer ObjectBuffer 
{
   ObjectData objectData[ ];
};

// Output
layout (location = 0) out vec3 outPos;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec3 outColor;
layout (location = 3) out vec2 outUV;
layout (location = 4) out vec4 outTangent;
layout (location = 5) out mat3 outTBN;
layout (location = 8) flat out uint outIndex;

void main() 
{
	outPos = inPos.xyz;
	outNormal = inNormal;
	outColor = inColor;
	outUV = inUV;
	outTangent = inTangent;
	outIndex = gl_InstanceIndex;

	vec3 N = normalize(inNormal);
	vec3 T = normalize(inTangent.xyz);
	vec3 B = cross(inNormal, inTangent.xyz) * inTangent.w;
	outTBN = mat3(T, B, N);
}