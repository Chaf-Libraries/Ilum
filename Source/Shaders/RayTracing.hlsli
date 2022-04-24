#ifndef __RAYTRACING_HLSL__
#define __RAYTRACING_HLSL__

#include "Math.hlsli"
#include "BxDF.hlsli"
#include "Material.hlsli"
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
[[vk::binding(12)]] StructuredBuffer<AreaLight> area_lights;
[[vk::binding(13)]] TextureCube Skybox;
[[vk::binding(13)]] SamplerState SkyboxSampler;

[[vk::push_constant]]
struct
{
    uint anti_alias;
    uint directional_light_count;
    uint spot_light_count;
    uint point_light_count;
    uint area_light_count;
    int max_bounce;
    float firefly_clamp_threshold;
    float parameter;
} push_constants;

struct RayPayload
{
    Interaction isect;
    Sampler _sampler;
        
    uint material_idx;
    
    float3 radiance;
    float3 throughout;
    
    float3 wi;
    float eta;
    
    bool visibility;
    bool terminate;
    bool ishit;
};

static const uint LightFlag_None = 1;
static const uint LightFlag_DeltaPosition = 1 << 1;
static const uint LightFlag_DeltaDirection = 1 << 2;

static const uint LightType_None = 0;
static const uint LightType_Directional = 1;
static const uint LightType_Point = 2;
static const uint LightType_Spot = 3;
static const uint LightType_Area = 3;

struct Light
{
    uint idx;
    
    bool IsIntersect(RayDesc ray, out float3 p)
    {
        uint count = push_constants.directional_light_count + push_constants.spot_light_count + push_constants.point_light_count;
        
        // Area Light
        if (idx >= count)
        {
            return area_lights[idx - count].IsIntersect(ray, p);
        }
        count -= push_constants.point_light_count;
        // Point Light
        if (idx >= count)
        {
            return point_lights[idx - count].IsIntersect(ray, p);
        }
        count -= push_constants.spot_light_count;
        // Spot Light
        if (idx >= count)
        {
            return spot_lights[idx - count].IsIntersect(ray, p);
        }
        count -= push_constants.directional_light_count;
        // Directional Light
        return directional_lights[idx - count].IsIntersect(ray, p);
        return false;
    }
    
    float3 Le(float3 p)
    {
        uint count = push_constants.directional_light_count + push_constants.spot_light_count + push_constants.point_light_count;
        
        // Area Light
        if (idx >= count)
        {
            AreaLight light = area_lights[idx - count];
            
            float3 tex_color = float3(1.0, 1.0, 1.0);
            if (light.tex_id < MAX_TEXTURE_ARRAY_SIZE)
            {
                float3 x_axis = (light.corners[1] - light.corners[0]).xyz;
                float3 y_axis = (light.corners[3] - light.corners[0]).xyz;
                
                float a = length(x_axis);
                float b = length(y_axis);
                            
                float2 uv = float2(dot(p - light.corners[0].xyz, normalize(x_axis)), dot(p - light.corners[0].xyz, normalize(y_axis)));
                uv = float2(uv.x / a, 1.0 - uv.y / b);
                tex_color = textureArray[NonUniformResourceIndex(light.tex_id)].SampleLevel(texSampler, uv, 0.0).rgb;
            }
                                   
            return light.Le() * tex_color;
        }
        count -= push_constants.point_light_count;
        // Point Light
        if (idx >= count)
        {
            return point_lights[idx - count].Le();
        }
        count -= push_constants.spot_light_count;
        // Spot Light
        if (idx >= count)
        {
            return spot_lights[idx - count].Le();
        }
        count -= push_constants.directional_light_count;
        // Directional Light
        return directional_lights[idx - count].Le();
        return float3(0.0, 0.0, 0.0);
    }
    
    float Area()
    {
        uint count = push_constants.directional_light_count + push_constants.spot_light_count + push_constants.point_light_count;
        
        // Area Light
        if (idx >= count)
        {
            return area_lights[idx - count].Area();

        }
        count -= push_constants.point_light_count;
        // Point Light
        if (idx >= count)
        {
            return point_lights[idx - count].Area();
        }
        count -= push_constants.spot_light_count;
        // Spot Light
        if (idx >= count)
        {
            return spot_lights[idx - count].Area();
        }
        count -= push_constants.directional_light_count;
        // Directional Light
        return directional_lights[idx - count].Area();
    }
    
    float3 SampleLi(Interaction interaction, float2 u, out float3 wi, out float pdf, out VisibilityTester visibility)
    {
        uint count = push_constants.directional_light_count + push_constants.spot_light_count + push_constants.point_light_count;
        
        // Area Light
        if (idx >= count)
        {
            AreaLight light = area_lights[idx - count];
            
            float3 tex_color = float3(1.0, 1.0, 1.0);
            if(light.tex_id<MAX_TEXTURE_ARRAY_SIZE)
            {
                tex_color = textureArray[NonUniformResourceIndex(light.tex_id)].SampleLevel(texSampler, u, 0.0).rgb;
            }
            
            return tex_color * area_lights[idx - count].SampleLi(interaction, u, wi, pdf, visibility);
        }
        count -= push_constants.point_light_count;
        // Point Light
        if (idx >= count)
        {
            return point_lights[idx - count].SampleLi(interaction, u, wi, pdf, visibility);
        }
        count -= push_constants.spot_light_count;
        // Spot Light
        if (idx >= count)
        {
            return spot_lights[idx - count].SampleLi(interaction, u, wi, pdf, visibility);
        }
        count -= push_constants.directional_light_count;
        // Directional Light
        return directional_lights[idx - count].SampleLi(interaction, u, wi, pdf, visibility);
    }
    
    float Power()
    {
        uint count = push_constants.directional_light_count + push_constants.spot_light_count + push_constants.point_light_count;
        
        // Area Light
        if (idx >= count)
        {
            return area_lights[idx - count].Power();
        }
        count -= push_constants.point_light_count;
        // Point Light
        if (idx >= count)
        {
            return point_lights[idx - count].Power();
        }
        count -= push_constants.spot_light_count;
        // Spot Light
        if (idx >= count)
        {
            return spot_lights[idx - count].Power();
        }
        count -= push_constants.directional_light_count;
        // Directional Light
        return directional_lights[idx - count].Power();
    }
    
    bool IsDelta()
    {
        uint count = push_constants.directional_light_count + push_constants.spot_light_count + push_constants.point_light_count;
        
        // Area Light
        if (idx >= count)
        {
            return area_lights[idx - count].IsDelta();
        }
        count -= push_constants.point_light_count;
        // Point Light
        if (idx >= count)
        {
            return point_lights[idx - count].IsDelta();
        }
        count -= push_constants.spot_light_count;
        // Spot Light
        if (idx >= count)
        {
            return spot_lights[idx - count].IsDelta();
        }
        count -= push_constants.directional_light_count;
        // Directional Light
        return directional_lights[idx - count].IsDelta();
    }
};

void GetInteraction(inout RayPayload ray_payload, RayDesc ray, BuiltInTriangleIntersectionAttributes Attributes)
{
    const uint instance_id = InstanceIndex();
    const uint primitive_id = PrimitiveIndex();
    const float3 bary = float3(1.0 - Attributes.barycentrics.x - Attributes.barycentrics.y, Attributes.barycentrics.x, Attributes.barycentrics.y);

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
    const float3 world_position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    // Tex Coord
    const float2 uv0 = v0.uv.xy;
    const float2 uv1 = v1.uv.xy;
    const float2 uv2 = v2.uv.xy;
    const float2 texcoord0 = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
    
	// Normal
    float3 nrm0 = v0.normal.xyz;
    float3 nrm1 = v1.normal.xyz;
    float3 nrm2 = v2.normal.xyz;
    float3 normal = normalize(nrm0 * bary.x + nrm1 * bary.y + nrm2 * bary.z);
    
    float3 world_normal = normalize(mul(WorldToObject4x3(), normal).xyz);
    float3 geom_normal = normalize(cross(pos2 - pos0, pos1 - pos0));
    float3 wgeom_normal = normalize(mul(WorldToObject4x3(), geom_normal).xyz);
    
    if (dot(ray_payload.isect.normal, wgeom_normal) <= 0)
    {
        ray_payload.isect.normal *= -1.0f;
    }
    
    ray_payload.isect.position = world_position;
    ray_payload.isect.normal = world_normal;
    ray_payload.isect.texCoord = texcoord0;
    ray_payload.isect.ffnormal = dot(ray_payload.isect.normal, ray.Direction) <= 0.0 ? ray_payload.isect.normal : -ray_payload.isect.normal;
    ray_payload.isect.CreateCoordinateSystem();
    ray_payload.isect.wo = -ray.Direction;
    ray_payload.material_idx = InstanceIndex();


}

void GetMaterial(inout Interaction interaction, out Material mat, RayDesc r, uint matID)
{
    MaterialData material = materials[matID];
    
    mat.base_color = material.base_color;
    mat.emissive = material.emissive_color * material.emissive_intensity;
    mat.subsurface = material.subsurface;
    mat.metallic = material.metallic;
    mat.specular = material.specular;
    mat.specular_tint = material.specular_tint;
    mat.roughness = material.roughness;
    mat.anisotropic = material.anisotropic;
    mat.sheen = material.sheen;
    mat.sheen_tint = material.sheen_tint;
    mat.clearcoat = material.clearcoat;
    mat.clearcoat_gloss = material.clearcoat_gloss;
    mat.specular_transmission = material.specular_transmission;
    mat.diffuse_transmission = material.diffuse_transmission;
    mat.refraction = material.refraction;
    mat.flatness = material.flatness;
    mat.thin = material.thin;
    mat.material_type = material.material_type;
    mat.data = material.data;

    if (material.textures[TEXTURE_BASE_COLOR] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float3 base_color = textureArray[NonUniformResourceIndex(material.textures[TEXTURE_BASE_COLOR])].SampleLevel(texSampler, interaction.texCoord, 0.0).rgb;
        base_color = pow(base_color, float3(2.2, 2.2, 2.2));
        mat.base_color.rgb *= base_color;
    }
    
    if (material.textures[TEXTURE_EMISSIVE] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float3 emissive = textureArray[NonUniformResourceIndex(material.textures[TEXTURE_EMISSIVE])].SampleLevel(texSampler, interaction.texCoord, 0.0).rgb;
        emissive = pow(emissive, float3(2.2, 2.2, 2.2));
        mat.emissive *= emissive;
    }
   
    if (material.textures[TEXTURE_METALLIC] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float metallic = textureArray[NonUniformResourceIndex(material.textures[TEXTURE_METALLIC])].SampleLevel(texSampler, interaction.texCoord, 0.0).r;
        mat.metallic *= metallic;
    }
     
    if (material.textures[TEXTURE_ROUGHNESS] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float roughness = textureArray[NonUniformResourceIndex(material.textures[TEXTURE_ROUGHNESS])].SampleLevel(texSampler, interaction.texCoord, 0.0).g;
        mat.roughness *= roughness;
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

bool SceneIntersection(RayDesc ray, inout RayPayload ray_payload)
{
    ray_payload.ishit = false;
        
    TraceRay(
        topLevelAS, // RaytracingAccelerationStructure
        RAY_FLAG_NONE, // RayFlags
        0xFF, // InstanceInclusionMask
        0, // RayContributionToHitGroupIndex
        1, // MultiplierForGeometryContributionToHitGroupIndex
        0, // MissShaderIndex
        ray, // Ray
        ray_payload // Payload
    );
    
    return ray_payload.ishit;
}

bool Unoccluded(inout RayPayload ray_payload, VisibilityTester visibility)
{
    ray_payload.visibility = false;
    RayDesc shadow_ray;
    shadow_ray.Direction = normalize(visibility.dir);
    shadow_ray.Origin = OffsetRay(visibility.from.position, dot(shadow_ray.Direction, visibility.from.normal) > 0 ? visibility.from.normal : -visibility.from.normal);
    shadow_ray.TMin = 0.0;
    shadow_ray.TMax = visibility.dist;
    
    TraceRay(
        topLevelAS, // RaytracingAccelerationStructure
        RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, // RayFlags
        0xFF, // InstanceInclusionMask
        0, // RayContributionToHitGroupIndex
        1, // MultiplierForGeometryContributionToHitGroupIndex
        materials[ray_payload.material_idx].material_type, // MissShaderIndex
        shadow_ray, // Ray
        ray_payload // Payload
    );
    
    return ray_payload.visibility;
}

// Environment Sampling (HDR)
// See:  https://arxiv.org/pdf/1901.05423.pdf
float3 EnvironmentSampling(RayPayload ray_payload, float3 wi, out float pdf)
{
    pdf = 1.0;
    return Skybox.SampleLevel(SkyboxSampler, wi, 0.0).rgb;
}

#endif