#ifndef _RAYTRACING_H
#define _RAYTRACING_H

//#include "BxDF.glsl"
#include "GlobalBuffer.glsl"
#include "Material.glsl"
#include "Math.glsl"
#include "Random.glsl"
#include "Sampling.glsl"

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 0, binding = 1, rgba16f) writeonly uniform image2D Image;
layout(set = 0, binding = 2, rgba16f) readonly uniform image2D PrevImage;

layout(set = 0, binding = 3) uniform CameraBuffer
{
	CameraData camera_data;
};

layout(set = 0, binding = 4) readonly buffer Vertices
{
	Vertex vertices[];
};

layout(set = 0, binding = 5) readonly buffer Indices
{
	uint indices[];
};

layout(set = 0, binding = 6) buffer PerInstanceBuffer
{
	PerInstanceData instance_data[];
};

layout(set = 0, binding = 7) buffer MaterialBuffer
{
	MaterialData material_data[];
};

layout(set = 0, binding = 8) uniform sampler2D texture_array[];

layout(set = 0, binding = 9) buffer DirectionalLights
{
	DirectionalLight directional_lights[];
};

layout(set = 0, binding = 10) buffer PointLights
{
	PointLight point_lights[];
};

layout(set = 0, binding = 11) buffer SpotLights
{
	SpotLight spot_lights[];
};

layout(set = 0, binding = 12) uniform samplerCube Skybox;

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

void CreateCoordinateSystem(in vec3 N, out vec3 Nt, out vec3 Nb)
{
	// http://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Vectors.html#CoordinateSystemfromaVector
	//if(abs(N.x) > abs(N.y))
	//  Nt = vec3(-N.z, 0, N.x) / sqrt(N.x * N.x + N.z * N.z);
	//else
	//  Nt = vec3(0, N.z, -N.y) / sqrt(N.y * N.y + N.z * N.z);
	//Nb = cross(N, Nt);

	Nt = normalize(((abs(N.z) > 0.99999f) ? vec3(-N.x * N.y, 1.0f - N.y * N.y, -N.y * N.z) :
                                            vec3(-N.x * N.z, -N.y * N.z, 1.0f - N.z * N.z)));
	Nb = cross(Nt, N);
}

ShadeState GetShadeState(in RayPayload ray_payload)
{
	ShadeState sstate;

	const uint instance_id  = ray_payload.instanceID;
	const uint primitive_id = ray_payload.primitiveID;
	const vec3 bary         = vec3(1.0 - ray_payload.baryCoord.x - ray_payload.baryCoord.y, ray_payload.baryCoord.x, ray_payload.baryCoord.y);

	PerInstanceData instance = instance_data[instance_id];

	// Vertex Attribute of the triangle
	Vertex v0 = vertices[instance.vertex_offset + indices[instance.index_offset + 3 * primitive_id]];
	Vertex v1 = vertices[instance.vertex_offset + indices[instance.index_offset + 3 * primitive_id + 1]];
	Vertex v2 = vertices[instance.vertex_offset + indices[instance.index_offset + 3 * primitive_id + 2]];

	const uint matIndex = instance_id;

	// Vertex of the triangle
	const vec3 pos0           = v0.position.xyz;
	const vec3 pos1           = v1.position.xyz;
	const vec3 pos2           = v2.position.xyz;
	const vec3 position       = pos0 * bary.x + pos1 * bary.y + pos2 * bary.z;
	const vec3 world_position = vec3(ray_payload.objectToWorld * vec4(position, 1.0));

	// Normal
	vec3 nrm0         = v0.normal.xyz;
	vec3 nrm1         = v1.normal.xyz;
	vec3 nrm2         = v2.normal.xyz;
	vec3 normal       = normalize(nrm0 * bary.x + nrm1 * bary.y + nrm2 * bary.z);
	vec3 world_normal = normalize(vec3(normal * ray_payload.worldToObject));
	vec3 geom_normal  = normalize(cross(pos1 - pos0, pos2 - pos0));
	vec3 wgeom_normal = normalize(vec3(geom_normal * ray_payload.worldToObject));

	// Tangent and Binormal
	/* const vec3 tng0     = v0.tangent.xyz;
	const vec3 tng1     = v1.tangent.xyz;
	const vec3 tng2     = v2.tangent.xyz;
	vec3       tangent  = (tng0.xyz * bary.x + tng1.xyz * bary.y + tng2.xyz * bary.z);
	tangent.xyz         = normalize(tangent.xyz);
	vec3 world_tangent  = normalize(vec3(mat4(ray_payload.objectToWorld) * vec4(tangent.xyz, 0)));
	world_tangent       = normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);
	vec3 world_binormal = normalize(cross(world_normal, world_tangent));*/
	vec3 world_tangent;
	vec3 world_binormal;
	CreateCoordinateSystem(world_normal, world_tangent, world_binormal);

	// Tex Coord
	const vec2 uv0       = v0.texcoord.xy;
	const vec2 uv1       = v1.texcoord.xy;
	const vec2 uv2       = v2.texcoord.xy;
	const vec2 texcoord0 = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

	sstate.normal      = world_normal;
	sstate.geom_normal = wgeom_normal;
	sstate.position    = world_position;
	sstate.tex_coord   = texcoord0;
	sstate.tangent_u   = world_tangent;
	sstate.tangent_v   = world_binormal;
	sstate.matIndex    = matIndex;

	// Move normal to same side as geometric normal
	if (dot(sstate.normal, sstate.geom_normal) <= 0)
	{
		sstate.normal *= -1.0f;
	}

	return sstate;
}

void GetMaterial(inout Interaction interaction, in Ray r)
{
	MaterialData material = material_data[interaction.matID];

	interaction.mat.base_color      = material.base_color;
	interaction.mat.emissive        = material.emissive_color * material.emissive_intensity;
	interaction.mat.subsurface      = material.subsurface;
	interaction.mat.metallic        = material.metallic;
	interaction.mat.specular        = material.specular;
	interaction.mat.specular_tint   = material.specular_tint;
	interaction.mat.roughness       = material.roughness;
	interaction.mat.anisotropic     = material.anisotropic;
	interaction.mat.sheen           = material.sheen;
	interaction.mat.sheen_tint      = material.sheen_tint;
	interaction.mat.clearcoat       = material.clearcoat;
	interaction.mat.clearcoat_gloss = material.clearcoat_gloss;
	interaction.mat.transmission    = material.transmission;
	interaction.mat.material_type   = material.material_type;
	interaction.mat.data            = material.data;

	if (material.textures[TEXTURE_BASE_COLOR] < MAX_TEXTURE_ARRAY_SIZE)
	{
		interaction.mat.base_color *= vec4(pow(texture(texture_array[nonuniformEXT(material.textures[TEXTURE_BASE_COLOR])], interaction.texCoord).rgb, vec3(2.2)), 1.0);
	}

	if (material.textures[TEXTURE_EMISSIVE] < MAX_TEXTURE_ARRAY_SIZE)
	{
		interaction.mat.emissive *= pow(texture(texture_array[nonuniformEXT(material.textures[TEXTURE_EMISSIVE])], interaction.texCoord).rgb, vec3(2.2));
	}

	if (material.textures[TEXTURE_METALLIC] < MAX_TEXTURE_ARRAY_SIZE)
	{
		interaction.mat.metallic *= texture(texture_array[nonuniformEXT(material.textures[TEXTURE_METALLIC])], interaction.texCoord).r;
	}

	if (material.textures[TEXTURE_ROUGHNESS] < MAX_TEXTURE_ARRAY_SIZE)
	{
		interaction.mat.roughness *= texture(texture_array[nonuniformEXT(material.textures[TEXTURE_ROUGHNESS])], interaction.texCoord).g;
	}

	if (material.textures[TEXTURE_NORMAL] < MAX_TEXTURE_ARRAY_SIZE)
	{
		mat3 TBN             = mat3(interaction.tangent, interaction.bitangent, interaction.normal);
		vec3 normalVector    = texture(texture_array[nonuniformEXT(material.textures[TEXTURE_NORMAL])], interaction.texCoord).rgb;
		normalVector         = normalize(normalVector * 2.0 - 1.0);
		interaction.normal   = normalize(TBN * normalVector);
		interaction.ffnormal = dot(interaction.normal, r.direction) <= 0.0 ? interaction.normal : -interaction.normal;
		CreateCoordinateSystem(interaction.ffnormal, interaction.tangent, interaction.bitangent);
	}
}

#endif