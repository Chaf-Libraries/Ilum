struct PerInstanceData
{
	mat4 world_transform;
	mat4 last_world_transform;
	mat4 pre_transform;

	vec3 bbox_min;
	uint entity_id;

	vec3 bbox_max;
};

struct CameraData 
{
    mat4 view_projection;
	mat4 last_view_projection;
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
    uint frustum_enable;
    uint backface_enable;
    
    uint occlusion_enable;
    uint meshlet_count;
    uint instance_count;
};

struct CountData
{
	uint visible_count;
    uint instance_visible_count;
    uint invisible_count;
    uint total_count;
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