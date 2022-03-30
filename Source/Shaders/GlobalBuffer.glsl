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
	uint frame_num;
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
#define BxDF_Matte 2
#define BxDF_Plastic 3
#define BxDF_Metal 4
#define BxDF_Mirror 5

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

struct Material
{
	vec4 base_color;
	vec3 emissive;
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
	uint  material_type;
};

// Light Source

struct DirectionalLight
{
	vec4  split_depth;
	mat4  view_projection[4];
	vec3  color;
	float intensity;
	vec3  direction;

	////
	// Rasterization Shadow
	int   shadow_mode;        // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS
	float filter_scale;
	int   filter_sample;
	int   sample_method;        // 0 - Uniform, 1 - Poisson Disk
	float light_size;
	////

	vec3 position;
};

struct PointLight
{
	vec3  color;
	float intensity;
	vec3  position;
	float constant;
	float linear_;
	float quadratic;

	// Rasterization Shadow
	int   shadow_mode;        // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS
	float filter_scale;
	int   filter_sample;
	int   sample_method;        // 0 - Uniform, 1 - Poisson Disk
	float light_size;
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

	// Rasterization Shadow
	int   shadow_mode;        // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS
	float filter_scale;
	int   filter_sample;
	int   sample_method;        // 0 - Uniform, 1 - Poisson Disk
	float light_size;
};

// Ray
struct Ray
{
	vec3 origin;
	vec3 direction;
	float tmin;
	float tmax;
};

struct Vertex
{
	vec4 position;
	vec4 texcoord;
	vec4 normal;
	vec4 tangent;
	vec4 bitangent;
};

struct RayPayload
{
	uint   seed;
	float  hitT;
	int    primitiveID;
	int    instanceID;
	vec2   baryCoord;
	mat4x3 objectToWorld;
	mat4x3 worldToObject;
};

struct ShadowPayload
{
	bool visibility;
};

struct ShadeState
{
	vec3 normal;
	vec3 geom_normal;
	vec3 position;
	vec2 tex_coord;
	vec3 tangent_u;
	vec3 tangent_v;
	uint matIndex;
};

struct Interaction
{
	int   depth;
	float eta;

	vec3 position;
	vec3 normal;
	vec3 ffnormal;
	vec3 tangent;
	vec3 bitangent;
	vec2 texCoord;

	uint     matID;
	Material mat;
};

struct BxdfSampleRec
{
	vec3  L;
	vec3  f;
	float pdf;
};

struct LightSampleRec
{
	vec3  surfacePos;
	vec3  normal;
	vec3  emission;
	float pdf;
};

#endif