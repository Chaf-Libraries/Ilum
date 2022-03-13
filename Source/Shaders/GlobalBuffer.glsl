#ifndef _GLOBAL_BUFFER_GLSL_
#define _GLOBAL_BUFFER_GLSL_

struct CameraData
{
	mat4 view_projection;
	mat4 last_view_projection;
	mat4 view_inverse;
	mat4 projection_inverse;
	vec4 frustum[6];
	vec3 position;
};

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

struct PerMeshletData
{
	uint instance_id;
	uint vertex_offset;
	uint index_offset;
	uint index_count;

	vec3  center;
	float radius;

	vec3  cone_apex;
	float cone_cutoff;

	vec3 cone_axis;
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
	uint  meshlet_count;
	uint  instance_count;
};

struct DrawIndexedIndirectCommand
{
	uint indexCount;
	uint instanceCount;
	uint firstIndex;
	int  vertexOffset;
	uint firstInstance;
};

struct CountData
{
	uint actual_draw;
	uint total_draw;
	uint meshlet_visible_count;
	uint instance_visible_count;
	uint meshlet_invisible_count;
	uint instance_invisible_count;
	uint meshlet_total_count;
	uint instance_total_count;
};

#define TEXTURE_BASE_COLOR 0
#define TEXTURE_NORMAL 1
#define TEXTURE_METALLIC 2
#define TEXTURE_ROUGHNESS 3
#define TEXTURE_EMISSIVE 4
#define TEXTURE_AO 5
#define TEXTURE_DISPLACEMENT 6
#define TEXTURE_MAX_NUM 7

#define BxDF_CookTorrance 0
#define BxDF_Disney 1

#define MAX_TEXTURE_ARRAY_SIZE 1024

struct MaterialData
{
	vec4 base_color;

	vec3  emissive_color;
	float emissive_intensity;

	float displacement;
	float subsurface;
	float metallic;
	float specular;

	float specular_tint;
	float roughness;
	float anisotropic;
	float sheen;

	float sheen_tint;
	float clearcoat;
	float clearcoat_gloss;
	float transmission;

	float transmission_roughness;
	uint  textures[TEXTURE_MAX_NUM];

	uint material_type;
};

struct DirectionalLight
{
	mat4  view_projection;
	vec3  color;
	float intensity;
	vec3  direction;
};

struct PointLight
{
	mat4  view_projection;
	vec3  color;
	float intensity;
	vec3  position;
	float constant;
	float linear;
	float quadratic;
};

struct SpotLight
{
	mat4  view_projection;
	vec3  color;
	float intensity;
	vec3  position;
	float cut_off;
	vec3  direction;
	float outer_cut_off;
};

#endif