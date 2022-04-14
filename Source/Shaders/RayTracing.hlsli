#ifndef __RAYTRACING_HLSL__
#define __RAYTRACING_HLSL__

#include "Math.hlsli"
#include "BxDF.hlsli"
//#include "Material.hlsli"
#include "Common.hlsli"
#include "Light.hlsli"
#include "Random.hlsli"

[[vk::binding(0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1)]] RWTexture2D<float4> Image;
[[vk::binding(2)]] RWTexture2D<float4> PrevImage;
[[vk::binding(3)]] ConstantBuffer<Camera> camera;
[[vk::binding(4)]] StructuredBuffer<Vertex> vertices;
[[vk::binding(5)]] StructuredBuffer<uint> indices;
[[vk::binding(6)]] StructuredBuffer<Instance> instances;
[[vk::binding(7)]] StructuredBuffer<MaterialData> materials;
[[vk::binding(8)]] Texture2D textureArray[];
[[vk::binding(8)]] SamplerState texSampler;
[[vk::binding(9)]] StructuredBuffer<DirectionalLight> directional_lights;
[[vk::binding(10)]] StructuredBuffer<PointLight> point_lights;
[[vk::binding(11)]] StructuredBuffer<SpotLight> spot_lights;
[[vk::binding(12)]] TextureCube Skybox;
[[vk::binding(12)]] SamplerState SkyboxSampler;

[[vk::push_constant]]
struct
{
    uint anti_alias;
    uint directional_light_count;
    uint spot_light_count;
    uint point_light_count;
    int max_bounce;
    float firefly_clamp_threshold;
    float parameter;
} push_constants;

float3 OffsetRay(float3 p, float3 n)
{
    const float intScale = 256.0f;
    const float floatScale = 1.0f / 65536.0f;
    const float origin = 1.0f / 32.0f;

    int3 of_i = int3(intScale * n.x, intScale * n.y, intScale * n.z);

    float3 p_i = float3(asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
	                asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
	                asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return float3(abs(p.x) < origin ? p.x + floatScale * n.x : p_i.x,
	            abs(p.y) < origin ? p.y + floatScale * n.y : p_i.y,
	            abs(p.z) < origin ? p.z + floatScale * n.z : p_i.z);
}

ShadeState GetShadeState(RayPayload ray_payload)
{
    ShadeState sstate;

    const uint instance_id = ray_payload.instanceID;
    const uint primitive_id = ray_payload.primitiveID;
    const float3 bary = float3(1.0 - ray_payload.baryCoord.x - ray_payload.baryCoord.y, ray_payload.baryCoord.x, ray_payload.baryCoord.y);

    Instance instance = instances[instance_id];

	// Vertex Attribute of the triangle
    Vertex v0 = vertices[instance.vertex_offset + indices[instance.index_offset + 3 * primitive_id]];
    Vertex v1 = vertices[instance.vertex_offset + indices[instance.index_offset + 3 * primitive_id + 1]];
    Vertex v2 = vertices[instance.vertex_offset + indices[instance.index_offset + 3 * primitive_id + 2]];

    const uint matIndex = instance_id;

	// Vertex of the triangle
    const float3 pos0 = v0.position.xyz;
    const float3 pos1 = v1.position.xyz;
    const float3 pos2 = v2.position.xyz;
    const float3 position = pos0 * bary.x + pos1 * bary.y + pos2 * bary.z;
    const float3 world_position = mul(float4(position, 1.0),ray_payload.objectToWorld).xyz;

	// Normal
    float3 nrm0 = v0.normal.xyz;
    float3 nrm1 = v1.normal.xyz;
    float3 nrm2 = v2.normal.xyz;
    float3 normal = normalize(nrm0 * bary.x + nrm1 * bary.y + nrm2 * bary.z);
    
    float3 world_normal = normalize(mul(ray_payload.worldToObject, normal).xyz);
    float3 geom_normal = normalize(cross(pos2 - pos0, pos1 - pos0));
    float3 wgeom_normal = normalize(mul(ray_payload.worldToObject, geom_normal).xyz);

	// Tangent and Binormal
    float3 world_tangent;
    float3 world_binormal;
    CreateCoordinateSystem(world_normal, world_tangent, world_binormal);

	// Tex Coord
    const float2 uv0 = v0.uv.xy;
    const float2 uv1 = v1.uv.xy;
    const float2 uv2 = v2.uv.xy;
    const float2 texcoord0 = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

    sstate.normal = world_normal;
    sstate.geom_normal = wgeom_normal;
    sstate.position = world_position;
    sstate.tex_coord = texcoord0;
    sstate.tangent_u = world_tangent;
    sstate.tangent_v = world_binormal;
    sstate.matIndex = matIndex;

	// Move normal to same side as geometric normal
    if (dot(sstate.normal, sstate.geom_normal) <= 0)
    {
        sstate.normal *= -1.0f;
    }

    return sstate;
}

void GetMaterial(inout Interaction interaction, RayDesc r, uint matID)
{
    MaterialData material = materials[matID];
    
    interaction.material.base_color = material.base_color;
    interaction.material.emissive = material.emissive_color * material.emissive_intensity;
    interaction.material.subsurface = material.subsurface;
    interaction.material.metallic = material.metallic;
    interaction.material.specular = material.specular;
    interaction.material.specular_tint = material.specular_tint;
    interaction.material.roughness = material.roughness;
    interaction.material.anisotropic = material.anisotropic;
    interaction.material.sheen = material.sheen;
    interaction.material.sheen_tint = material.sheen_tint;
    interaction.material.clearcoat = material.clearcoat;
    interaction.material.clearcoat_gloss = material.clearcoat_gloss;
    interaction.material.specular_transmission = material.specular_transmission;
    interaction.material.diffuse_transmission = material.diffuse_transmission;
    interaction.material.refraction = material.refraction;
    interaction.material.flatness = material.flatness;
    interaction.material.thin = material.thin;
    interaction.material.material_type = material.material_type;
    interaction.material.data = material.data;

    if (material.textures[TEXTURE_BASE_COLOR] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float3 base_color = textureArray[NonUniformResourceIndex(material.textures[TEXTURE_BASE_COLOR])].SampleLevel(texSampler, interaction.texCoord, 0.0).rgb;
        base_color = pow(base_color, float3(2.2, 2.2, 2.2));
        interaction.material.base_color.rgb *= base_color;
    }
    
    if (material.textures[TEXTURE_EMISSIVE] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float3 emissive = textureArray[NonUniformResourceIndex(material.textures[TEXTURE_EMISSIVE])].SampleLevel(texSampler, interaction.texCoord, 0.0).rgb;
        emissive = pow(emissive, float3(2.2, 2.2, 2.2));
        interaction.material.emissive *= emissive;
    }
   
    if (material.textures[TEXTURE_METALLIC] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float metallic = textureArray[NonUniformResourceIndex(material.textures[TEXTURE_METALLIC])].SampleLevel(texSampler, interaction.texCoord, 0.0).r;
        interaction.material.metallic *= metallic;
    }
     
    if (material.textures[TEXTURE_ROUGHNESS] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float roughness = textureArray[NonUniformResourceIndex(material.textures[TEXTURE_ROUGHNESS])].SampleLevel(texSampler, interaction.texCoord, 0.0).g;
        interaction.material.roughness *= roughness;
    }
    
    if (material.textures[TEXTURE_NORMAL] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float3x3 TBN = float3x3(interaction.tangent, interaction.bitangent, interaction.normal);
        float3 normalVector = textureArray[NonUniformResourceIndex(material.textures[TEXTURE_NORMAL])].SampleLevel(texSampler, interaction.texCoord, 0.0).rgb;
        normalVector = normalize(normalVector * 2.0 - 1.0);
        interaction.normal = normalize(mul(normalVector, TBN));
        interaction.ffnormal = dot(interaction.normal, r.Direction) <= 0.0 ? interaction.normal : -interaction.normal;
        interaction.CreateCoordinateSystem();
    }
}

bool SceneIntersection(RayDesc ray, inout Interaction isect)
{
    RayPayload payload;
    payload.hitT = Infinity;
        
    TraceRay(
        topLevelAS, // RaytracingAccelerationStructure
        RAY_FLAG_NONE, // RayFlags
        0xFF, // InstanceInclusionMask
        0, // RayContributionToHitGroupIndex
        1, // MultiplierForGeometryContributionToHitGroupIndex
        0, // MissShaderIndex
        ray, // Ray
        payload // Payload
    );
    
    ShadeState sstate = GetShadeState(payload);
    
    isect.position = sstate.position;
    isect.normal = sstate.normal;
    isect.tangent = sstate.tangent_u;
    isect.bitangent = sstate.tangent_v;
    isect.texCoord = sstate.tex_coord;
    isect.ffnormal = dot(isect.normal, ray.Direction) <= 0.0 ? isect.normal : -isect.normal;
    isect.wo = -ray.Direction;
    
    GetMaterial(isect, ray, sstate.matIndex);

    return payload.hitT != Infinity;
}

bool VisibilityTest(Interaction from, float3 dir, float dist)
{
    ShadowPayload payload;
    payload.visibility = false;
    RayDesc shadow_ray;
    shadow_ray.Direction = normalize(dir);
    shadow_ray.Origin = OffsetRay(from.position, dot(shadow_ray.Direction, from.normal) > 0 ? from.normal : -from.normal);
    shadow_ray.TMin = 0.0;
    shadow_ray.TMax = dist;
    
    TraceRay(
        topLevelAS, // RaytracingAccelerationStructure
        RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, // RayFlags
        0xFF, // InstanceInclusionMask
        0, // RayContributionToHitGroupIndex
        1, // MultiplierForGeometryContributionToHitGroupIndex
        1, // MissShaderIndex
        shadow_ray, // Ray
        payload // Payload
    );
    
    return payload.visibility;
}

// Environment Sampling (HDR)
// See:  https://arxiv.org/pdf/1901.05423.pdf


#endif