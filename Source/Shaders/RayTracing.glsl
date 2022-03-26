#ifndef _RAYTRACING_H
#define _RAYTRACING_H

#include "Geometry.glsl"
#include "GlobalBuffer.glsl"
#include "Interaction.glsl"
#include "Random.glsl"
#include "Sampling.glsl"

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 0, binding = 1, rgba16f) uniform image2D Image;

layout(set = 0, binding = 2) uniform CameraBuffer
{
	CameraData camera_data;
};

layout(set = 0, binding = 3) readonly buffer Vertices
{
	Vertex vertices[];
};

layout(set = 0, binding = 4) readonly buffer Indices
{
	uint indices[];
};

layout(set = 0, binding = 5) buffer PerInstanceBuffer
{
	PerInstanceData instance_data[];
};

layout(set = 0, binding = 6) buffer MaterialBuffer
{
	MaterialData material_data[];
};

layout(set = 0, binding = 7) uniform sampler2D texture_array[];

layout(set = 0, binding = 8) buffer DirectionalLights
{
	DirectionalLight directional_lights[];
};

layout(set = 0, binding = 9) buffer PointLights
{
	PointLight point_lights[];
};

layout(set = 0, binding = 10) buffer SpotLights
{
	SpotLight spot_lights[];
};

layout(set = 0, binding = 11) uniform samplerCube Skybox;

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

Ray CameraCastRay(CameraData camera, vec2 screen_coords)
{
	Ray ray;

	vec4 target = camera.projection_inverse * vec4(screen_coords.x, screen_coords.y, 1, 1);

	ray.origin    = (camera.view_inverse * vec4(0, 0, 0, 1)).xyz;
	ray.direction = (camera.view_inverse * vec4(normalize(target.xyz), 0)).xyz;
	ray.tmin      = 0.0;
	ray.tmax      = Infinity;

	return ray;
}

vec3 OffsetRay(in vec3 p, in vec3 n)
{
	const float intScale   = 256.0f;
	const float floatScale = 1.0f / 65536.0f;
	const float origin     = 1.0f / 32.0f;

	ivec3 of_i = ivec3(intScale * n.x, intScale * n.y, intScale * n.z);

	vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
	                intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
	                intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

	return vec3(abs(p.x) < origin ? p.x + floatScale * n.x : p_i.x,
	            abs(p.y) < origin ? p.y + floatScale * n.y : p_i.y,
	            abs(p.z) < origin ? p.z + floatScale * n.z : p_i.z);
}


Interaction GetInteraction(in RayPayload prd)
{
	Interaction interaction;

	const vec3 bary         = vec3(1.0 - prd.baryCoord.x - prd.baryCoord.y, prd.baryCoord.x, prd.baryCoord.y);
	const uint instance_id  = prd.instanceID;
	const uint primitive_id = prd.primitiveID;

	PerInstanceData instance = instance_data[instance_id];

	// Vertex Attribute of the triangle
	Vertex v0 = vertices[instance.vertex_offset + indices[instance.index_offset + 3 * primitive_id]];
	Vertex v1 = vertices[instance.vertex_offset + indices[instance.index_offset + 3 * primitive_id + 1]];
	Vertex v2 = vertices[instance.vertex_offset + indices[instance.index_offset + 3 * primitive_id + 2]];

	// Vertex of the triangle
	const vec3 pos0           = v0.position.xyz;
	const vec3 pos1           = v1.position.xyz;
	const vec3 pos2           = v2.position.xyz;
	const vec3 position       = pos0 * bary.x + pos1 * bary.y + pos2 * bary.z;
	const vec3 world_position = vec3(prd.objectToWorld * vec4(position, 1.0));

	// Normal of the triangle
	const vec3 nrm0         = v0.normal.xyz;
	const vec3 nrm1         = v1.normal.xyz;
	const vec3 nrm2         = v2.normal.xyz;
	const vec3 normal       = normalize(nrm0 * bary.x + nrm1 * bary.y + nrm2 * bary.z);
	const vec3 world_normal = normalize(vec3(normal * prd.worldToObject));
	const vec3 geom_normal  = normalize(cross(pos2 - pos0, pos1 - pos0));
	const vec3 wgeom_normal = normalize(vec3(geom_normal * prd.worldToObject));

	// Tangent of the triangle
	const vec3 tng0          = v0.tangent.xyz;
	const vec3 tng1          = v1.tangent.xyz;
	const vec3 tng2          = v2.tangent.xyz;
	const vec3 tangent       = normalize(tng0 * bary.x + tng1 * bary.y + tng2 * bary.z);
	const vec3 world_tangent = normalize(vec3(tangent * prd.worldToObject));

	// Bitangent of the triangle
	const vec3 btng0           = v0.bitangent.xyz;
	const vec3 btng1           = v1.bitangent.xyz;
	const vec3 btng2           = v2.bitangent.xyz;
	const vec3 bitangent       = normalize(btng0 * bary.x + btng1 * bary.y + btng2 * bary.z);
	const vec3 world_bitangent = normalize(vec3(bitangent * prd.worldToObject));

	// Texcoord
	const vec2 uv0      = v0.texcoord.xy;
	const vec2 uv1      = v1.texcoord.xy;
	const vec2 uv2      = v2.texcoord.xy;
	const vec2 texcoord = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

	interaction.normal      = world_normal;
	interaction.geom_normal = wgeom_normal;
	interaction.position    = world_position;
	interaction.texcoord    = texcoord;
	interaction.tangent     = world_tangent;
	interaction.bitangent   = world_bitangent;
	interaction.material    = material_data[instance_id];

	if (dot(interaction.normal, interaction.geom_normal) <= 0)
	{
		interaction.normal *= -1.0f;
	}

	return interaction;
}

#endif