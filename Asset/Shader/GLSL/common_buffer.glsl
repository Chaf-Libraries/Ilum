#ifndef COMMON_BUFFER_GLSL
#define COMMON_BUFFER_GLSL

struct PerInstanceData
{
	mat4 transform;
	mat4 last_transform;

	vec3 bbox_min;
	uint entity_id;

	vec3 bbox_max;
	uint material_id;

	uint vertex_offset;
	uint index_offset;
	uint index_count;
};

struct CameraData 
{
    mat4 view_projection;
	mat4 last_view_projection;
	mat4 view_inverse;
	mat4 projection_inverse;
	vec4 frustum[6];
	vec3 position;
};

struct CullingData
{
	mat4 view;

    mat4 last_view;

    float P00;
    float P11;
    float znear;
    float zfar;

    float zbuffer_width;
    float zbuffer_height;
    uint meshlet_count;
    uint instance_count;
};

struct CountData
{
	uint actual_draw;
	uint meshlet_visible_count;
    uint instance_visible_count;
    uint meshlet_invisible_count;
    uint instance_invisible_count;
    uint meshlet_total_count;
    uint instance_total_count;
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

struct DrawIndexedIndirectCommand 
{
	uint indexCount;
	uint instanceCount;
	uint firstIndex;
	int vertexOffset;
	uint firstInstance;
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

#endif