#version 450

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_shader_draw_parameters: enable

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

layout (set = 1, binding = 0) uniform sampler2D textureArray[];

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inColor;
layout (location = 3) in vec2 inUV;
layout (location = 4) in vec3 inViewVec;
layout (location = 5) in vec3 inLightVec;
layout (location = 6) in vec4 inTangent;
layout (location = 7) in mat3 inTBN;
layout (location = 10) flat in uint inIndex;

layout (location = 0) out vec4 outFragColor;


void main() 
{
	//texture(textures[nonuniformEXT(inTexIndex)], inUV);
	// vec4 color = texture(textureArray[nonuniformEXT(textureIndex.baseColorTextureIndex)], inUV) * vec4(inColor, 1.0);
vec4 color = texture(textureArray[nonuniformEXT(objectData[inIndex].baseColorTextureIndex)], inUV) * vec4(inColor, 1.0);

	if (objectData[inIndex].alphaMode == 1) {
		if (color.a < objectData[inIndex].alphaCutOff) {
			discard;
		}
	}

	vec3 N = normalize(inNormal);
	vec3 T = normalize(inTangent.xyz);
	vec3 B = cross(inNormal, inTangent.xyz) * inTangent.w;
	mat3 TBN = mat3(T, B, N);
	N = TBN * normalize(texture(textureArray[nonuniformEXT(objectData[inIndex].normalTextureIndex)], inUV).xyz * 2.0 - vec3(1.0));

	const float ambient = 0.1;
	vec3 L = normalize(inLightVec);
	vec3 V = normalize(inViewVec);
	vec3 R = reflect(-L, N);
	vec3 diffuse = max(dot(N, L), ambient).rrr;
	float specular = pow(max(dot(R, V), 0.0), 32.0);
	outFragColor = vec4(diffuse * color.rgb, color.a);
	// float depth=texture(depthMap, inUV).x;
	// outFragColor = vec4(vec3(depth), 1.0);


}