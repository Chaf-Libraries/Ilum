
#ifndef SHADER_INTEROP_H
#define SHADER_INTEROP_H
#ifdef __cplusplus
#	include "../Dependencies/cereal/include/cereal/cereal.hpp"
#	include "../Dependencies/glm/glm/glm.hpp"

namespace ShaderInterop
{
#endif        // __cplusplus

#define MAX_TEXTURE_NUM 1024

#ifdef __cplusplus
using float2   = glm::vec2;
using float3   = glm::vec3;
using float4   = glm::vec4;
using uint     = uint32_t;
using uint2    = glm::uvec2;
using uint3    = glm::uvec3;
using uint4    = glm::uvec4;
using int2     = glm::ivec2;
using int3     = glm::ivec3;
using int4     = glm::ivec4;
using float4x4 = glm::mat4;
#	define asint(x) *(int *) (&x)
#	define asuint(x) *(uint32_t *) (&x)
#	define asfloat(x) *(float *) (&x)

inline uint firstbithigh(uint value)
{
	uint bit = 0;

	uint start = 1 << (31 - bit);
	while ((value & start) == 0 && start != 1)
	{
		start = 1 << (31 - ++bit);
	}
	return bit;
}

#endif

static const uint MESHLET_MAX_TRIANGLES = 124;
static const uint MESHLET_MAX_VERTICES  = 64;

struct Vertex
{
	float4 position;
	float4 texcoord;
	float4 normal;
	float4 tangent;

#ifdef __cplusplus
	template <typename Archive>
	void serialize(Archive ar)
	{
		ar(position, texcoord, normal, tangent);
	}
#endif
};

struct Material
{
	uint type;

	// PBR Specular Glossiness
	float3 pbr_specular_factor;
	float4 pbr_diffuse_factor;
	float  pbr_glossiness_factor;
	uint   pbr_diffuse_texture;
	uint   pbr_specular_glossiness_texture;

	// PBR Metallic Roughness
	float  pbr_metallic_factor;
	float4 pbr_base_color_factor;
	float  pbr_roughness_factor;
	uint   pbr_base_color_texture;
	uint   pbr_metallic_roughness_texture;

	// Emissive
	float  emissive_strength;
	float3 emissive_factor;
	uint   emissive_texture;

	// Sheen
	float3 sheen_color_factor;
	float  sheen_roughness_factor;
	uint   sheen_texture;
	uint   sheen_roughness_texture;

	// Clear Coat
	float clearcoat_factor;
	float clearcoat_roughness_factor;
	uint  clearcoat_texture;
	uint  clearcoat_roughness_texture;
	uint  clearcoat_normal_texture;

	// Specular
	float  specular_factor;
	float3 specular_color_factor;
	uint   specular_texture;
	uint   specular_color_texture;

	// Transmission
	float transmission_factor;
	uint  transmission_texture;

	// Volume
	float  attenuation_distance;
	float  thickness_factor;
	float3 attenuation_color;
	uint   thickness_texture;

	// Iridescence
	float iridescence_factor;
	float iridescence_ior;
	float iridescence_thickness_min;
	float iridescence_thickness_max;
	uint  iridescence_thickness_texture;
	uint  iridescence_texture;

	// IOR
	float ior;

	float alpha_cut_off;
	uint  alpha_mode;

	uint normal_texture;
	uint occlusion_texture;

	uint unlit;
	uint double_sided;
};

struct MeshletBound
{
	float3 center;
	float  radius;
	float3 cone_axis;
	float  cone_cut_off;
};

struct Meshlet
{
	uint         vertex_offset;
	uint         primitive_offset;
	uint         vertex_count;
	uint         primitive_count;
	MeshletBound bound;

#ifdef __cplusplus
	template <typename Archive>
	void serialize(Archive ar)
	{
		ar(vertex_offset, primitive_offset, vertex_count, primitive_count,
		   bound.center, bound.cone_axis, bound.cone_cut_off, bound.radius);
	}
#endif
};

struct Camera
{
	float4x4 view;
	float4x4 projection;
	float4x4 view_projection;
	float3   position;
	uint     frame_count;
	float4   right;
	float4   up;

#ifndef __cplusplus
	/* RayDesc CastRay(float2 screen_coords)
	{
	    RayDesc ray;
	    float4  target = mul(inv_projection, float4(screen_coords.x, screen_coords.y, 1, 1));
	    ray.Origin     = mul(inv_view, float4(0, 0, 0, 1)).xyz;
	    ray.Direction  = mul(inv_view, float4(normalize(target.xyz), 0)).xyz;
	    ray.TMin       = 0.0;
	    ray.TMax       = Infinity;
	    return ray;
	}*/
#endif        // !__cplusplus
};

struct Instance
{
	float4x4 transform;
	float3   aabb_min;
	uint     material;
	float3   aabb_max;
	uint     mesh;
	uint     meshlet_count;
	uint     id;
};

struct DirectionalLight
{
	float4   split_depth;
	float4x4 view_projection[4];
	float4   shadow_cam_pos[4];
	float3   color;
	float    intensity;
	float3   direction;
};

struct SpotLight
{
	float4x4 view_projection;
	float3   color;
	float    intensity;
	float3   position;
	float    cut_off;
	float    outer_cut_off;
	float3   direction;
};

struct PointLight
{
	float3 color;
	float  intensity;
	float3 position;
	float  range;
};

struct AreaLight
{
	float3 color;
	float  intensity;

	float4 corners[4];
};

struct SceneInfo
{
	float3 aabb_min;
	uint   directional_light_count;

	float3 aabb_max;
	uint   point_light_count;

	uint spot_light_count;
	uint area_light_count;
	uint instance_count;
	uint meshlet_count;

	uint vertices_count;
	uint primitives_count;
};

struct HierarchyNode
{
	uint parent;
	uint left_child;
	uint right_child;
};

struct AABB
{
	float4 min_val;
	float4 max_val;

#ifndef __cplusplus
	AABB Transform(float4x4 trans)
	{
		trans = transpose(trans);

		float3 v[2], xa, xb, ya, yb, za, zb;

		xa = trans[0].xyz * min_val[0];
		xb = trans[0].xyz * max_val[0];

		ya = trans[1].xyz * min_val[1];
		yb = trans[1].xyz * max_val[1];

		za = trans[2].xyz * min_val[2];
		zb = trans[2].xyz * max_val[2];

		v[0] = trans[3].xyz;
		v[0] += min(xa, xb);
		v[0] += min(ya, yb);
		v[0] += min(za, zb);

		v[1] = trans[3].xyz;
		v[1] += max(xa, xb);
		v[1] += max(ya, yb);
		v[1] += max(za, zb);

		AABB aabb;
		aabb.min_val = float4(v[0], 0.0);
		aabb.max_val = float4(v[1], 0.0);

		return aabb;
	}
#endif
};

#ifdef __cplusplus
inline uint PackTriangle(uint8_t v0, uint8_t v1, uint8_t v2)
#else
int  PackTriangle(uint v0, uint v1, uint v2)
#endif
{
	uint encode = 0;
	encode += v0 & 0xff;
	encode += (v1 & 0xff) << 8;
	encode += (v2 & 0xff) << 16;
	return encode;
}

#ifdef __cplusplus
inline void UnPackTriangle(uint encode, uint8_t &v0, uint8_t &v1, uint8_t &v2)
#else
void UnPackTriangle(uint encode, out uint v0, out uint v1, out uint v2)
#endif
{
#ifdef __cplusplus
	uint8_t a = encode & 0xff;
	uint8_t b = (encode >> 8) & 0xff;
	uint8_t c = (encode >> 16) & 0xff;
#else
	uint a = encode & 0xff;
	uint b = (encode >> 8) & 0xff;
	uint c = (encode >> 16) & 0xff;
#endif

	v0 = a;
	v1 = b;
	v2 = c;
}

inline uint PackVBuffer(uint instance_id, uint meshlet_id, uint primitive_id)
{
	// Primitive ID 7
	// Meshlet ID 11
	// Instance ID 14
	uint vbuffer = 0;
	vbuffer += primitive_id & 0x7f;
	vbuffer += (meshlet_id & 0x3ff) << 7;
	vbuffer += (instance_id & 0x3fff) << 18;
	return vbuffer;
}

#ifdef __cplusplus
inline void UnPackVBuffer(uint vbuffer, uint &instance_id, uint &meshlet_id, uint &primitive_id)
#else
void UnPackVBuffer(uint vbuffer, out uint instance_id, out uint meshlet_id, out uint primitive_id)
#endif
{
	// Primitive ID 7
	// Meshlet ID 11
	// Instance ID 14
	primitive_id = vbuffer & 0x7f;
	meshlet_id   = (vbuffer >> 7) & 0x3ff;
	instance_id  = (vbuffer >> 18) & 0x3fff;
}

#ifdef __cplusplus
}
#endif

#endif