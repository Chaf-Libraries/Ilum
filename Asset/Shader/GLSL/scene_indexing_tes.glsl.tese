#version 450 core                                                                             

layout(triangles, fractional_odd_spacing, cw) in;       

// Output
struct PosPatch
{
    vec3 pos[9];
};

layout (location = 0) in patch PosPatch inPosPatch;
layout (location = 9) in vec3 inNormal[];
layout (location = 10) in vec3 inColor[];
layout (location = 11) in vec2 inUV[];
layout (location = 12) in vec4 inTangent[];
layout (location = 13) in mat3 inTBN[];
layout (location = 16) flat in uint inIndex[];

layout (location = 0) out vec3 outPos;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec3 outColor;
layout (location = 3) out vec2 outUV;
layout (location = 4) out vec3 outViewVec;
layout (location = 5) out vec3 outLightVec;
layout (location = 6) out vec4 outTangent;
layout (location = 7) out mat3 outTBN;
layout (location = 10) out uint outIndex;

// Uniform buffer
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

vec3 interpolation(vec3 v1, vec3 v2, vec3 v3)
{
    return gl_TessCoord.x * v1 + gl_TessCoord.y * v2 + gl_TessCoord.z * v3;
}

vec2 interpolation(vec2 v1, vec2 v2, vec2 v3)
{
    return gl_TessCoord.x * v1 + gl_TessCoord.y * v2 + gl_TessCoord.z * v3;
}

vec4 interpolation(vec4 v1, vec4 v2, vec4 v3)
{
    return gl_TessCoord.x * v1 + gl_TessCoord.y * v2 + gl_TessCoord.z * v3;
}

mat3 interpolation(mat3 v1, mat3 v2, mat3 v3)
{
    return gl_TessCoord.x * v1 + gl_TessCoord.y * v2 + gl_TessCoord.z * v3;
}

void main()                                                                                    
{                                                                                               
  
    float u = gl_TessCoord.x;                                                                 
    float v = gl_TessCoord.y;                                                                  
    float w = gl_TessCoord.z;

    outNormal = interpolation(inNormal[0], inNormal[1], inNormal[2]);
    outColor = interpolation(inColor[0], inColor[1], inColor[2]);
    outUV = interpolation(inUV[0], inUV[1], inUV[2]);
    outTangent = interpolation(inTangent[0], inTangent[1], inTangent[2]);
    outTBN = interpolation(inTBN[0], inTBN[1], inTBN[2]);
    outIndex = inIndex[2];
  
    vec3 N = normalize(outNormal);
	vec3 T = normalize(outTangent.xyz);
	vec3 B = cross(outNormal, outTangent.xyz) * outTangent.w;
	outTBN = mat3(T, B, N);

    
    vec3 P1= inPosPatch.pos[0] * u + inPosPatch.pos[1] * v + inPosPatch.pos[2] * w;
    vec3 P2= inPosPatch.pos[3] * u + inPosPatch.pos[4] * v + inPosPatch.pos[5] * w;
    vec3 P3= inPosPatch.pos[6] * u + inPosPatch.pos[7] * v + inPosPatch.pos[8] * w;

    outPos = P1 * u+ P2 * v + P3 * w;

    //outPos = P1;  
    gl_Position = vec4(outPos, 1.0);
	gl_Position = uboScene.projection * uboScene.view * objectData[outIndex].model* gl_Position;
    gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;
	
    //outNormal = mat3(primitive.model) * outNormal;
	vec4 pos = objectData[outIndex].model * vec4(outPos, 1.0);
	outLightVec = uboScene.lightPos.xyz - pos.xyz;
	outViewVec = uboScene.viewPos.xyz - pos.xyz;
}  