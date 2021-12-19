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
layout(location = 6) out uint outMeshletIndex;

struct PerInstanceData
{
	mat4 world_transform;
	mat4 pre_transform;

	vec3 bbox_min;
	uint entity_id;

	vec3 bbox_max;
};

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

	uint displacement_map;
	float displacement_height;
};

struct PerMeshletData
{
	// Vertex
	uint instance_id;
	uint vertex_offset;
	uint index_offset;
	uint index_count;

	vec3 center;
	float radius;

	vec3 cone_apex;
	float cone_cutoff;

	vec3 cone_axis;
};

layout (set = 0, binding = 0) uniform MainCamera
{
    mat4 view_projection;
	mat4 last_view_projection;
	vec4 frustum[6];
	vec3 position;
}main_camera;

layout (set = 0, binding = 1) uniform sampler2D textureArray[];

layout (set = 0, binding = 2) buffer PerInstanceBuffer
{
    PerInstanceData instance_data[];
};

layout (set = 0, binding = 3) buffer MaterialBuffer
{
    MaterialData material_data[];
};

layout (set = 0, binding = 4) buffer PerMeshletBuffer
{
    PerMeshletData meshlet_data[];
};

layout (set = 0, binding = 5) buffer DrawBuffer
{
    uint draw_data[];
};



void main() {
    outIndex = draw_data[gl_DrawIDARB];
    outMeshletIndex = gl_DrawIDARB;

    float height = material_data[outIndex].displacement_map < 1024?
        max(textureLod(textureArray[nonuniformEXT(material_data[outIndex].displacement_map)], inUV, 0.0).r, 0.0) * material_data[outIndex].displacement_height:
        0.0;
    
    mat4 trans = instance_data[outIndex].world_transform * instance_data[outIndex].pre_transform;

    // World normal
    mat3 mNormal = transpose(inverse(mat3(trans)));
    outNormal = mNormal * normalize(inNormal);
    outTangent = mNormal * normalize(inTangent);
    outBiTangent = mNormal * normalize(inBiTangent);

    outPos = trans * vec4(inPos, 1.0);

    outPos.xyz += material_data[outIndex].displacement_map < 1024?
        normalize(outNormal) * (max(textureLod(textureArray[nonuniformEXT(material_data[outIndex].displacement_map)], inUV, 0.0).r, 0.0) * material_data[outIndex].displacement_height):
        vec3(0.0);

    gl_Position = main_camera.view_projection * outPos;

    outPos.w = gl_Position.z;

    outUV = inUV;
}