#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "../GlobalFunction.glsl"
#include "../GlobalBuffer.glsl"
#include "RayTracing.glsl"

struct Vertex
{
	vec4 position;
	vec4 texcoord;
	vec4 normal;
	vec4 tangent;
	vec4 bitangent;
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
	PerInstanceData instance_data[ ];
};

layout (set = 0, binding = 6) buffer MaterialBuffer
{
    MaterialData material_data[ ];
};

layout (set = 0, binding = 7) uniform sampler2D texture_array[];

layout(set = 0, binding = 8) buffer DirectionalLights{
    DirectionalLight directional_lights[ ];
};

layout(set = 0, binding = 9) buffer PointLights{
    PointLight point_lights[ ];
};

layout(set = 0, binding = 10) buffer SpotLights{
    SpotLight spot_lights[ ];
};

layout(push_constant) uniform PushBlock{
    uint directional_light_count;
    uint spot_light_count;
    uint point_light_count;
}push_data;

layout(location = 0) rayPayloadInEXT vec3 hitValue;

hitAttributeEXT vec2 attribs;

void main()
{
	const vec3 barycentricCoords = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

	PerInstanceData instance = instance_data[gl_InstanceID];
	MaterialData material = material_data[gl_InstanceID];

	Vertex v0 = vertices[instance.vertex_offset + indices[instance.index_offset + 3 * gl_PrimitiveID]];
	Vertex v1 = vertices[instance.vertex_offset + indices[instance.index_offset + 3 * gl_PrimitiveID + 1]];
	Vertex v2 = vertices[instance.vertex_offset + indices[instance.index_offset + 3 * gl_PrimitiveID + 2]];

	const vec3 pos      = v0.position.xyz * barycentricCoords.x + v1.position.xyz * barycentricCoords.y + v2.position.xyz * barycentricCoords.z;
	const vec3 world_pos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));

	const vec3 nrm      = v0.normal.xyz * barycentricCoords.x + v1.normal.xyz * barycentricCoords.y + v2.normal.xyz * barycentricCoords.z;
	const vec3 N = normalize(vec3(nrm * gl_WorldToObjectEXT));

	const vec3 W0 = -gl_WorldRayDirectionEXT;
	const vec3 R = reflect(-W0, N);







	vec3 radiance = vec3(0.0);
	for(uint i = 0; i< push_data.point_light_count; i++)
	{
		PointLight light = point_lights[i];
		vec3 L = normalize(light.position - world_pos);
		float d = length(light.position - world_pos);
		float NoL = max(0.0, dot(N,L));
		float Fatt = 1.0/(light.constant + light.linear_ * d + light.quadratic * d * d);
		radiance += light.color.rgb * light.intensity * Fatt;
	}

	if(material.textures[TEXTURE_BASE_COLOR] < 1024)
	{
		vec2 texCoord = v0.texcoord.xy * barycentricCoords.x + v1.texcoord.xy * barycentricCoords.y + v2.texcoord.xy * barycentricCoords.z;
		hitValue = texture(texture_array[nonuniformEXT(material.textures[TEXTURE_BASE_COLOR])], texCoord).xyz;
	}
	else
	{
		hitValue = material.base_color.rgb;
	}


	//hitValue = material.base_color.rgb;
	//hitValue = worldNrm;
}
