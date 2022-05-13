
#ifndef SHADER_INTEROP_H
#define SHADER_INTEROP_H
#ifdef __cplusplus
#	include "../Dependencies/cereal/include/cereal/cereal.hpp"
#	include "../Dependencies/glm/glm/glm.hpp"

namespace ShaderInterop
{
#endif        // __cplusplus

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
	float4 albedo_factor;

	float3 specular_factor;
	float  glossiness_factor;

	float metallic_factor;
	float roughness_factor;
	uint  type;
	uint  alpha_mode;

	float3 emissive_factor;
	float  emissive_strength;

	uint albedo_texture;
	uint normal_texture;
	uint emissive_texture;
	uint specular_glossiness_texture;

#ifdef __cplusplus
	alignas(16)
#endif
	    uint metallic_roughness_texture;
	float alpha_cut_off;
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
	float4x4 inv_view;
	float4x4 inv_projection;
	float4x4 view_projection;
	float3   position;
	uint     frame_count;

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
	uint     material;
	uint     mesh;
	uint     meshlet_count;
	uint     padding2;
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