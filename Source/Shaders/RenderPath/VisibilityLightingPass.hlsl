#include "../Common.hlsli"
#include "../Random.hlsli"
#include "../SphericalHarmonic.hlsli"

#define PREFILTER_MIP_LEVELS 5

#ifndef RUNTIME
#define HAS_MESH
#define HAS_SKINNED_MESH
#define HAS_POINT_LIGHT
#define HAS_SPOT_LIGHT
#define HAS_DIRECTIONAL_LIGHT
#define HAS_RECT_LIGHT
#define HAS_ENV_LIGHT
#define HAS_SHADOW_MAP
#define HAS_CASCADE_SHADOW_MAP
#define HAS_OMNI_SHADOW_MAP
#define HAS_IRRADIANCE_SH
#define HAS_PREFILTER_MAP
#define SHADOW_FILTER_NONE
#define SHADOW_FILTER_HARD
#define SHADOW_FILTER_PCF
#define SHADOW_FILTER_PCSS
#endif

Texture2D<uint> VisibilityBuffer;
Texture2D<float> DepthBuffer;

#ifdef HAS_MESH
StructuredBuffer<Instance> MeshInstanceBuffer;
#endif
#ifdef HAS_SKINNED_MESH
StructuredBuffer<Instance> SkinnedMeshInstanceBuffer;
#endif

RWStructuredBuffer<uint> MaterialCountBuffer;

[numthreads(8, 8, 1)]
void CollectMaterialCount(CSParam param)
{
    uint2 dispatch_id = param.DispatchThreadID.xy;
    
    uint2 extent;
    VisibilityBuffer.GetDimensions(extent.x, extent.y);
    
    if (dispatch_id.x >= extent.x ||
        dispatch_id.y >= extent.y ||
        DepthBuffer.Load(int3(dispatch_id, 0)).r > 3e38f)
    {
        return;
    }
    
    uint mesh_type, instance_id, primitive_id;
    UnPackVisibilityBuffer(VisibilityBuffer.Load(uint3(dispatch_id, 0)), mesh_type, instance_id, primitive_id);
    
    uint material_id = ~0;
#ifdef HAS_MESH
    if (mesh_type == MESH_TYPE)
    {
        material_id = MeshInstanceBuffer.Load(instance_id).material_id;
    }
#endif
#ifdef HAS_SKINNED_MESH
    if (mesh_type == SKINNED_MESH_TYPE)
    {
        material_id = SkinnedMeshInstanceBuffer.Load(instance_id).material_id;
    }
#endif
    
    if (material_id != ~0)
    {
        uint temp = 0;
        InterlockedAdd(MaterialCountBuffer[material_id], 1, temp);
    }
}

RWStructuredBuffer<uint> MaterialOffsetBuffer;

groupshared uint shared_data[256];

[numthreads(128, 1, 1)]
void CalculateMaterialOffset(CSParam param)
{
    uint id = param.GroupThreadID.x;
    
    uint material_count, temp;
    MaterialCountBuffer.GetDimensions(material_count, temp);
    
    uint range = (1u << (uint(log2(material_count)) + 1u));
    
    if (id < material_count)
    {
        shared_data[id] = MaterialCountBuffer[id];
    }
    else
    {
        shared_data[id] = 0;
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    if (id >= range - 1)
    {
        return;
    }
    
    uint steps = uint(log2(id + 1)) + 1u;
    
    for (uint step = 0; step < steps; step++)
    {
        uint rd_id = range - 2u - id;
        uint wd_id = range - 2u - id + (1u << step);
        shared_data[wd_id] += shared_data[rd_id];
        GroupMemoryBarrierWithGroupSync();
    }
    
    if (id < material_count - 1)
    {
        MaterialOffsetBuffer[id + 1] = shared_data[id];
    }
}

RWStructuredBuffer<DispatchIndirectCommand> IndirectCommandBuffer;
RWStructuredBuffer<uint> MaterialPixelBuffer;

[numthreads(8, 8, 1)]
void CalculatePixelBuffer(CSParam param)
{
    uint2 dispatch_id = param.DispatchThreadID.xy;
    
    uint2 extent;
    VisibilityBuffer.GetDimensions(extent.x, extent.y);
    
    if (DepthBuffer.Load(int3(dispatch_id, 0)).r >= 3e38f ||
        dispatch_id.x >= extent.x ||
        dispatch_id.y >= extent.y)
    {
        return;
    }
    
    uint mesh_type, instance_id, primitive_id;
    UnPackVisibilityBuffer(VisibilityBuffer.Load(uint3(dispatch_id, 0)), mesh_type, instance_id, primitive_id);
    
    uint material_id = ~0;
    
#ifdef HAS_MESH
    if (mesh_type == MESH_TYPE)
    {
        material_id = MeshInstanceBuffer.Load(instance_id).material_id;
    }
#endif
    
#ifdef HAS_SKINNED_MESH
    if (mesh_type == SKINNED_MESH_TYPE)
    {
        material_id = SkinnedMeshInstanceBuffer.Load(instance_id).material_id;
    }
#endif
    
    uint idx;
    InterlockedAdd(IndirectCommandBuffer[material_id].x, 1, idx);
    MaterialPixelBuffer[idx + MaterialOffsetBuffer[material_id]] = PackXY(dispatch_id.x, dispatch_id.y);
}

[numthreads(8, 1, 1)]
void CalculateIndirectArgument(CSParam param)
{
    uint dispatch_id = param.DispatchThreadID.x;
    
    uint material_count, temp;
    IndirectCommandBuffer.GetDimensions(material_count, temp);
    
    if (dispatch_id >= material_count)
    {
        return;
    }
    
    IndirectCommandBuffer[dispatch_id].x = (IndirectCommandBuffer[dispatch_id].x + 8 - 1) / 8;
    IndirectCommandBuffer[dispatch_id].y = 1;
    IndirectCommandBuffer[dispatch_id].z = 1;
}

#ifndef DISPATCH_INDIRECT
#define MATERIAL_ID 0
#include "../Material/Material.hlsli"
#endif

#include "../Light.hlsli"

RWTexture2D<float4> LightDirectIllumination;
RWTexture2D<float4> EnvDirectIllumination;
RWTexture2D<float4> PositionDepth;
RWTexture2D<float4> NormalRoughness;
RWTexture2D<float4> AlbedoMetallic;

ConstantBuffer<View> ViewBuffer;
ConstantBuffer<LightInfo> LightInfoBuffer;

#if defined(HAS_SHADOW_MAP) || defined(HAS_CASCADE_SHADOW_MAP) || defined(HAS_OMNI_SHADOW_MAP)
SamplerState ShadowMapSampler;
#endif

#ifdef HAS_POINT_LIGHT
StructuredBuffer<PointLight> PointLightBuffer;
#ifdef HAS_OMNI_SHADOW_MAP
TextureCubeArray<float> PointLightShadow;
#endif
#endif

#ifdef HAS_SPOT_LIGHT
StructuredBuffer<SpotLight> SpotLightBuffer;
#ifdef HAS_SHADOW_MAP
Texture2DArray<float> SpotLightShadow;
#endif
#endif

#ifdef HAS_DIRECTIONAL_LIGHT
StructuredBuffer<DirectionalLight> DirectionalLightBuffer;
#ifdef HAS_CASCADE_SHADOW_MAP
Texture2DArray<float> DirectionalLightShadow;
#endif
#endif

#ifdef HAS_RECT_LIGHT
StructuredBuffer<RectLight> RectLightBuffer;
#endif

#ifdef HAS_MESH
StructuredBuffer<Vertex> MeshVertexBuffer[];
StructuredBuffer<uint> MeshIndexBuffer[];
#endif

#ifdef HAS_SKINNED_MESH
StructuredBuffer<SkinnedVertex> SkinnedMeshVertexBuffer[];
StructuredBuffer<uint> SkinnedMeshIndexBuffer[];
StructuredBuffer<float4x4> BoneMatrices[];
#endif

#ifdef HAS_ENV_LIGHT
#ifdef HAS_IRRADIANCE_SH
RWTexture2D<float4> IrradianceSH;
#endif
#ifdef HAS_PREFILTER_MAP
SamplerState PrefilterMapSampler;
TextureCube<float4> PrefilterMap;
Texture2D<float2> GGXPreintegration;
#endif
#endif

uint hash(uint a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

float3 GetColor(uint v)
{
    uint mhash = hash(v);
    return float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
}

void CalculateBarycentre(float4x4 transform, float2 pixel, uint2 extent, float3 v0, float3 v1, float3 v2, out float3 bary, out float3 bary_ddx, out float3 bary_ddy)
{
    float4 clip_pos0 = mul(ViewBuffer.view_projection_matrix, mul(transform, float4(v0.xyz, 1.0)));
    float4 clip_pos1 = mul(ViewBuffer.view_projection_matrix, mul(transform, float4(v1.xyz, 1.0)));
    float4 clip_pos2 = mul(ViewBuffer.view_projection_matrix, mul(transform, float4(v2.xyz, 1.0)));
        
    float2 p0 = clip_pos1.xy - clip_pos0.xy;
    float2 p1 = clip_pos0.xy - clip_pos2.xy;
        
    // Calculate barycentric
    float3 inv_w = rcp(float3(clip_pos0.w, clip_pos1.w, clip_pos2.w));
        
    float3 ndc0 = clip_pos0.xyz / clip_pos0.w;
    float3 ndc1 = clip_pos1.xyz / clip_pos1.w;
    float3 ndc2 = clip_pos2.xyz / clip_pos2.w;
    float2 ndc = (float2(pixel) + float2(0.5, 0.5)) / float2(extent);
    ndc = ndc * 2 - 1.0;
    ndc.y *= -1;
        
    float3 pos_120_x = float3(ndc1.x, ndc2.x, ndc0.x);
    float3 pos_120_y = float3(ndc1.y, ndc2.y, ndc0.y);
    float3 pos_201_x = float3(ndc2.x, ndc0.x, ndc1.x);
    float3 pos_201_y = float3(ndc2.y, ndc0.y, ndc1.y);
        
    float3 C_dx = pos_201_y - pos_120_y;
    float3 C_dy = pos_120_x - pos_201_x;
     
    float3 C = C_dx * (ndc.x - pos_120_x) + C_dy * (ndc.y - pos_120_y);
    float3 G = C * inv_w;
    
    float H = dot(C, inv_w);
    float rcpH = rcp(H);
        
    bary = G * rcpH;
        
    float3 G_dx = C_dx * inv_w;
    float3 G_dy = C_dy * inv_w;

    float H_dx = dot(C_dx, inv_w);
    float H_dy = dot(C_dy, inv_w);
        
    bary_ddx = (G_dx * H - G * H_dx) * (rcpH * rcpH) * (2.0f / float(extent.x));
    bary_ddy = (G_dy * H - G * H_dy) * (rcpH * rcpH) * (-2.0f / float(extent.y));
}

#if defined(HAS_SHADOW_MAP) || defined(HAS_CASCADE_SHADOW_MAP) || defined(HAS_OMNI_SHADOW_MAP)
float LinearizeDepth(float depth, float znear, float zfar)
{
    float z = depth * 2.0 - 1.0;
    return znear * zfar / (zfar + depth * (znear - zfar));
}

float FindBlock(Texture2DArray<float> shadowmap, float4 shadow_coord, float layer, float filter_scale, uint filter_sample)
{
    uint2 tex_dim;
    uint layers;
    shadowmap.GetDimensions(tex_dim.x, tex_dim.y, layers);

	// Find blocker
    float z_blocker = 0.0;
    float num_blockers = 0.0;
    float2 offset = float2(0.0, 0.0);
    for (int i = 0; i < filter_sample; i++)
    {
        offset = PoissonDiskSamples2D(shadow_coord.xy + offset, filter_sample, 10, i);
        offset = offset * filter_scale / float2(tex_dim);
        float dist = shadowmap.SampleLevel(ShadowMapSampler, float3(shadow_coord.xy + offset, layer), 0.0).r;
        if (dist < shadow_coord.z)
        {
            num_blockers += 1.0;
            z_blocker += dist;
        }
    }

    if (num_blockers == 0.0)
    {
        return 0.0;
    }

    return num_blockers == 0.0 ? 0.0 : z_blocker / num_blockers;
}

float FindBlockCube(TextureCubeArray<float> shadowmap, float3 L, float layer, float filter_scale, uint filter_sample)
{
    uint2 tex_dim;
    uint layers;
    shadowmap.GetDimensions(tex_dim.x, tex_dim.y, layers);

    float light_depth = length(L);

	// Find blocker
    float z_blocker = 0.0;
    float num_blockers = 0.0;
    float disk_radius = filter_scale / float(max(tex_dim.x, tex_dim.y));
    float3 offset = float3(0.0, 0.0, 0.0);
    
    int x = int(sqrt(filter_sample));
    int y = filter_sample / x;
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            // Poisson sampling
            offset = PoissonDiskSamples3D(L + offset, x * y, 10, float2(i, j)) * disk_radius;
            float dist = shadowmap.SampleLevel(ShadowMapSampler, float4(L + offset, layer), 0.0).r;
            if (light_depth > dist)
            {
                num_blockers += 1.0;
                z_blocker += dist;
            }
        }
    }

    if (num_blockers == 0.0)
    {
        return 0.0;
    }

    return num_blockers == 0.0 ? 0.0 : z_blocker / num_blockers;
}

// Sample shadow map
float SampleShadowmap(Texture2DArray<float> shadowmap, float4 shadow_coord, float layer, float2 offset)
{
    float shadow = 1.0;

    if (shadow_coord.z > -1.0 && shadow_coord.z < 1.0)
    {
        float dist = shadowmap.SampleLevel(ShadowMapSampler, float3(shadow_coord.xy + offset, layer), 0.0).r;
        if (shadow_coord.w > 0.0 && dist < shadow_coord.z)
        {
            shadow = 0.0;
        }
    }
    return shadow;
}

// Sample shadow map via PCF
float SampleShadowmapPCF(Texture2DArray<float> shadowmap, float4 shadow_coord, float layer, uint filter_sample, float filter_scale)
{
    uint2 tex_dim;
    uint layers;
    shadowmap.GetDimensions(tex_dim.x, tex_dim.y, layers);

    float shadow_factor = 0.0;

    float2 offset = float2(0.0, 0.0);
    for (int i = 0; i < filter_sample; i++)
    {
        offset = PoissonDiskSamples2D(shadow_coord.xy + offset, filter_sample, 10, i);
        offset = offset * filter_scale / float2(tex_dim);
        shadow_factor += SampleShadowmap(shadowmap, shadow_coord, layer, offset);
    }

    return shadow_factor / float(filter_sample);
}

// Sample shadow map via PCSS
float SampleShadowmapPCSS(Texture2DArray<float> shadowmap, float4 shadow_coord, float layer, uint filter_sample, float filter_scale, float light_scale)
{
    uint2 tex_dim;
    uint layers;
    shadowmap.GetDimensions(tex_dim.x, tex_dim.y, layers);

    float z_receiver = shadow_coord.z * shadow_coord.w;

	// Penumbra size
    float z_blocker = LinearizeDepth(FindBlock(shadowmap, shadow_coord, layer, filter_scale, filter_sample), 0.01, 1000.0);
    float w_light = 0.1;
    float w_penumbra = (z_receiver - z_blocker) * light_scale / z_blocker;

	// Filtering
    float shadow_factor = 0.0;
    float2 offset = float2(0.0, 0.0);
    for (int i = 0; i < filter_sample; i++)
    {
        offset = PoissonDiskSamples2D(shadow_coord.xy + offset, filter_sample, 10, i);
        offset = offset * w_penumbra / float2(tex_dim);
        shadow_factor += SampleShadowmap(shadowmap, shadow_coord, layer, offset);
    }

    return shadow_factor / float(filter_sample);
}

// Sample shadow cubemap
float SampleShadowmapCube(TextureCubeArray<float> shadowmap, float3 L, float layer, float3 offset)
{
    float shadow = 1.0;
    float light_depth = length(L);
    L.z = -L.z;
	// Reconstruct depth
    float dist = shadowmap.SampleLevel(ShadowMapSampler, float4(L + offset, layer), 0.0).r;
    dist *= 100.0;

    if (light_depth > dist)
    {
        shadow = 0.0;
    }

    return shadow;
}

// Sample shadow cubemap via PCF
float SampleShadowmapCubePCF(TextureCubeArray<float> shadowmap, float3 L, float layer, float filter_scale, uint filter_sample)
{
    uint2 tex_dim;
    uint layers;
    shadowmap.GetDimensions(tex_dim.x, tex_dim.y, layers);

    float shadow_factor = 0.0;
    float light_depth = length(L);

    float disk_radius = filter_scale / float(max(tex_dim.x, tex_dim.y));

    float3 offset = float3(0.0, 0.0, 0.0);
    int count = 0;
    int x = int(sqrt(filter_sample));
    int y = filter_sample / x;
    count = x * y;
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            // Poisson sampling
            offset = PoissonDiskSamples3D(L + offset, count, 10, float2(i, j)) * disk_radius;
            shadow_factor += SampleShadowmapCube(shadowmap, L, layer, offset);
        }
    }
    return shadow_factor / float(count);
}

// Sample shadow cubemap via PCSS
float SampleShadowmapCubePCSS(TextureCubeArray<float> shadowmap, float3 L, float layer, float filter_scale, uint filter_sample, float light_scale)
{
    uint2 tex_dim;
    uint layers;
    shadowmap.GetDimensions(tex_dim.x, tex_dim.y, layers);

    float light_depth = length(L);
    float z_receiver = light_depth;

	// Penumbra size
    float z_blocker = LinearizeDepth(FindBlockCube(shadowmap, L, layer, filter_scale, filter_sample), 0.01, 1000.0);
    float w_light = 0.1;
    float w_penumbra = (z_receiver - z_blocker) * light_scale / z_blocker;

	// Filtering
    float shadow_factor = 0.0;
    float3 offset = float3(0.0, 0.0, 0.0);
    float disk_radius = filter_scale / float(max(tex_dim.x, tex_dim.y));
    int count = 0;
    int x = int(sqrt(filter_sample));
    int y = filter_sample / x;
    count = x * y;
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            // Poisson sampling
            offset = PoissonDiskSamples3D(L + offset, count, 10, float2(i, j)) * w_penumbra / float(tex_dim.x);
            shadow_factor += SampleShadowmapCube(shadowmap, L, layer, offset);
        }
    }
    return shadow_factor / float(filter_sample);
}
#endif

#ifdef HAS_POINT_LIGHT
float CalculatePointLightShadow(PointLight light, uint idx, SurfaceInteraction interaction)
{
    float3 L = interaction.isect.p - light.position;
    
#ifdef HAS_OMNI_SHADOW_MAP
#ifdef SHADOW_FILTER_NONE
    return 1.f;
#endif
    
#ifdef SHADOW_FILTER_HARD
    return SampleShadowmapCube(PointLightShadow, L, idx, float3(0.0, 0.0, 0.0));
#endif
    
#ifdef SHADOW_FILTER_PCF
    return SampleShadowmapCubePCF(PointLightShadow, L, idx, light.filter_scale, light.filter_sample);
#endif
    
#ifdef SHADOW_FILTER_PCSS
    return SampleShadowmapCubePCSS(PointLightShadow, L, idx, light.filter_scale, light.filter_sample, light.light_scale);
#endif
    
#else
    return 1.f;
#endif
}
#endif

#ifdef HAS_SPOT_LIGHT
float CalculateSpotLightShadow(SpotLight light, uint idx, SurfaceInteraction interaction)
{
    float4 shadow_clip = mul(light.view_projection, float4(interaction.isect.p, 1.0));
    float4 shadow_coord = float4(shadow_clip.xyz / shadow_clip.w, shadow_clip.w);
    shadow_coord.xy = shadow_coord.xy * 0.5 + 0.5;
    shadow_coord.y = 1.0 - shadow_coord.y;

#ifdef HAS_OMNI_SHADOW_MAP
#ifdef SHADOW_FILTER_NONE
    return 1.f;
#endif
    
#ifdef SHADOW_FILTER_HARD
    return SampleShadowmap(SpotLightShadow, shadow_coord, idx, float2(0.0, 0.0));
#endif
    
#ifdef SHADOW_FILTER_PCF
    return SampleShadowmapPCF(SpotLightShadow, shadow_coord, idx, light.filter_sample, light.filter_scale);
#endif
    
#ifdef SHADOW_FILTER_PCSS
    return SampleShadowmapPCSS(SpotLightShadow, shadow_coord, idx, light.filter_sample, light.filter_scale, light.light_scale);
#endif
    
#else
    return 1.f;
#endif
}
#endif

#ifdef HAS_DIRECTIONAL_LIGHT
float CalculateDirectionalLightShadow(DirectionalLight light, uint idx, SurfaceInteraction interaction, float linear_z)
{
    float shadow = 1.f;
    
    linear_z = mul(ViewBuffer.view_projection_matrix, float4(interaction.isect.p, 1.f)).z;
    
    uint cascade_index = 0;
	// Select cascade
    for (uint i = 0; i < 3; ++i)
    {
        if (light.split_depth[i] > -linear_z)
        {
            cascade_index = i + 1;
        }
    }

    float4 shadow_clip = mul(light.view_projection[cascade_index], float4(interaction.isect.p, 1.0));
    float4 shadow_coord = float4(shadow_clip.xyz / shadow_clip.w, shadow_clip.w);
    shadow_coord.xy = shadow_coord.xy * 0.5 + 0.5;
    shadow_coord.y = 1.0 - shadow_coord.y;

    uint layer = idx * 4 + cascade_index;
    
#ifdef HAS_CASCADE_SHADOW_MAP
#ifdef SHADOW_FILTER_NONE
    return 1.f;
#endif
    
#ifdef SHADOW_FILTER_HARD
    return SampleShadowmap(DirectionalLightShadow, shadow_coord, idx * 4 + cascade_index, float2(0.0, 0.0));
#endif
    
#ifdef SHADOW_FILTER_PCF
    return SampleShadowmapPCF(DirectionalLightShadow, shadow_coord, idx * 4 + cascade_index, light.filter_sample, light.filter_scale);
#endif
    
#ifdef SHADOW_FILTER_PCSS
    return SampleShadowmapPCSS(DirectionalLightShadow, shadow_coord, idx * 4 + cascade_index, light.filter_sample, light.filter_scale, light.light_scale);
#endif

#else
    return 1.f;
#endif
}
#endif

[numthreads(8, 1, 1)]
void DispatchIndirect(CSParam param)
{
    uint id = param.DispatchThreadID.x;
    uint offset = MaterialOffsetBuffer.Load(MATERIAL_ID);
    uint count = MaterialCountBuffer.Load(MATERIAL_ID);
    
    if (id >= count)
    {
        return;
    }
    
    uint2 extent;
    VisibilityBuffer.GetDimensions(extent.x, extent.y);
    
    uint2 pixel;
    UnpackXY(MaterialPixelBuffer.Load(id + offset), pixel.x, pixel.y);
    
    uint instance_id, primitive_id, mesh_type;
    UnPackVisibilityBuffer(VisibilityBuffer.Load(uint3(pixel, 0)), mesh_type, instance_id, primitive_id);
    
    SurfaceInteraction interaction;
    
    float3 bary, bary_ddx, bary_ddy;
    
    float linear_z = 0.f;
    
#ifdef HAS_MESH
    if (mesh_type == MESH_TYPE)
    {
        Instance instance = MeshInstanceBuffer.Load(instance_id);
        uint mesh_id = instance.mesh_id;
        
        Vertex v0 = MeshVertexBuffer[mesh_id][MeshIndexBuffer[mesh_id][primitive_id * 3]];
        Vertex v1 = MeshVertexBuffer[mesh_id][MeshIndexBuffer[mesh_id][primitive_id * 3 + 1]];
        Vertex v2 = MeshVertexBuffer[mesh_id][MeshIndexBuffer[mesh_id][primitive_id * 3 + 2]];
        
        CalculateBarycentre(instance.transform, pixel, extent, v0.position, v1.position, v2.position, bary, bary_ddx, bary_ddy);
        
        interaction.isect.p = v0.position.xyz * bary.x + v1.position.xyz * bary.y + v2.position.xyz * bary.z;
        linear_z = mul(ViewBuffer.view_matrix, mul(instance.transform, float4(interaction.isect.p, 1.0))).z;
        interaction.isect.p = mul(instance.transform, float4(interaction.isect.p, 1.0)).xyz;
        interaction.isect.uv = v0.texcoord0.xy * bary.x + v1.texcoord0.xy * bary.y + v2.texcoord0.xy * bary.z;
        interaction.isect.n = v0.normal.xyz * bary.x + v1.normal.xyz * bary.y + v2.normal.xyz * bary.z;
        interaction.isect.n = normalize(mul((float3x3) instance.transform, normalize(interaction.isect.n)));
        interaction.isect.nt = v0.tangent.xyz * bary.x + v1.tangent.xyz * bary.y + v2.tangent.xyz * bary.z;
        interaction.isect.nt = normalize(mul((float3x3) instance.transform, normalize(interaction.isect.nt)).xyz);
        interaction.isect.wo = normalize(ViewBuffer.position - interaction.isect.p);
        interaction.shading_n = dot(interaction.isect.n, interaction.isect.wo) <= 0 ? -interaction.isect.n : interaction.isect.n;
        interaction.isect.t = length(ViewBuffer.position - interaction.isect.p);
        interaction.material = instance.material_id;
        interaction.duvdx = v0.texcoord0.xy * bary_ddx.x + v1.texcoord0.xy * bary_ddx.y + v2.texcoord0.xy * bary_ddx.z;
        interaction.duvdy = v0.texcoord0.xy * bary_ddy.x + v1.texcoord0.xy * bary_ddy.y + v2.texcoord0.xy * bary_ddy.z;
    }
#endif
    
#ifdef HAS_SKINNED_MESH
    if (mesh_type == SKINNED_MESH_TYPE)
    {
        Instance instance = SkinnedMeshInstanceBuffer.Load(instance_id);
        uint mesh_id = instance.mesh_id;
        
        SkinnedVertex v[3];
        
        v[0] = SkinnedMeshVertexBuffer[mesh_id][SkinnedMeshIndexBuffer[mesh_id][primitive_id * 3]];
        v[1] = SkinnedMeshVertexBuffer[mesh_id][SkinnedMeshIndexBuffer[mesh_id][primitive_id * 3 + 1]];
        v[2] = SkinnedMeshVertexBuffer[mesh_id][SkinnedMeshIndexBuffer[mesh_id][primitive_id * 3 + 2]];
        
        float3 position[3];
        float3 normal[3];
        float3 tangent[3];
        
        if (instance.animation_id != ~0U)
        {
            uint bone_count = 0;
            uint bone_stride = 0;
            BoneMatrices[instance.animation_id].GetDimensions(bone_count, bone_stride);
            
            for (uint v_idx = 0; v_idx < 3; v_idx++)
            {
                position[v_idx] = 0.f;
                normal[v_idx] = 0.f;
                tangent[v_idx] = 0.f;
                
                for (uint bone_idx = 0; bone_idx < MAX_BONE_INFLUENCE; bone_idx++)
                {
                    int bone = v[v_idx].bones[bone_idx];
                    float weight = v[v_idx].weights[bone_idx];
            
                    if (bone == -1)
                    {
                        continue;
                    }
                    
                    if (bone >= bone_count)
                    {
                        position[v_idx] = v[v_idx].position;
                        normal[v_idx] = v[v_idx].normal;
                        tangent[v_idx] = v[v_idx].tangent;
                        break;
                    }

                    position[v_idx] += mul(BoneMatrices[instance.animation_id][bone], float4(v[v_idx].position, 1.0f)).xyz * weight;
                    normal[v_idx] += mul((float3x3) BoneMatrices[instance.animation_id][bone], v[v_idx].normal) * weight;
                    tangent[v_idx] += mul((float3x3) BoneMatrices[instance.animation_id][bone], v[v_idx].tangent) * weight;
                }
            }
        }
        else
        {
            for (uint v_idx = 0; v_idx < 3; v_idx++)
            {
                position[v_idx] = v[v_idx].position;
                normal[v_idx] = v[v_idx].normal;
                tangent[v_idx] = v[v_idx].tangent;
            }
        }
        
        CalculateBarycentre(instance.transform, pixel, extent, position[0], position[1], position[2], bary, bary_ddx, bary_ddy);
        
        interaction.isect.p = position[0].xyz * bary.x + position[1].xyz * bary.y + position[2].xyz * bary.z;
        linear_z = mul(ViewBuffer.view_matrix, mul(instance.transform, float4(interaction.isect.p, 1.0))).z;
        interaction.isect.p = mul(instance.transform, float4(interaction.isect.p, 1.0)).xyz;
        interaction.isect.uv = v[0].texcoord0.xy * bary.x + v[1].texcoord0.xy * bary.y + v[2].texcoord0.xy * bary.z;
        interaction.isect.n = normal[0].xyz * bary.x + normal[1].xyz * bary.y + normal[2].xyz * bary.z;
        interaction.isect.n = normalize(mul((float3x3) instance.transform, normalize(interaction.isect.n)));
        interaction.isect.nt = tangent[0].xyz * bary.x + tangent[1].xyz * bary.y + tangent[2].xyz * bary.z;
        interaction.isect.nt = normalize(mul((float3x3) instance.transform, normalize(interaction.isect.nt)).xyz);
        interaction.isect.wo = normalize(ViewBuffer.position - interaction.isect.p);
        interaction.shading_n = dot(interaction.isect.n, interaction.isect.wo) <= 0 ? -interaction.isect.n : interaction.isect.n;
        interaction.isect.t = length(ViewBuffer.position - interaction.isect.p);
        interaction.material = instance.material_id;
        interaction.duvdx = v[0].texcoord0.xy * bary_ddx.x + v[1].texcoord0.xy * bary_ddx.y + v[2].texcoord0.xy * bary_ddx.z;
        interaction.duvdy = v[0].texcoord0.xy * bary_ddy.x + v[1].texcoord0.xy * bary_ddy.y + v[2].texcoord0.xy * bary_ddy.z;
    }
#endif
    
    Material material;
    material.Init(interaction);
    
    GBufferData g_buffer_data = material.bsdf.GetGBufferData();
    
    float3 radiance = g_buffer_data.emissive;
    float3 ambient = 0.f;
    
    LightSampleContext li_ctx;
    li_ctx.n = interaction.isect.n;
    li_ctx.ns = interaction.shading_n;
    li_ctx.p = interaction.isect.p;
    
    // Handle point light DI
#ifdef HAS_POINT_LIGHT
    uint point_light_id = 0;
    for (uint i = 0; i < LightInfoBuffer.point_light_count; i++)
    {
        PointLight light = PointLightBuffer.Load(i);
        LightLiSample light_sample = light.SampleLi(li_ctx, 0.f);
        float3 f = material.bsdf.Eval(interaction.isect.wo, light_sample.wi, TransportMode_Radiance);
        float3 color = f * light_sample.L;
        if (light.cast_shadow)
        {
            color *= CalculatePointLightShadow(light, light.shadow_id, interaction);
            point_light_id++;
        }
        radiance += color;

    }
#endif
    
    // Handle spot light DI
#ifdef HAS_SPOT_LIGHT
    uint spot_light_id = 0;
    for (uint i = 0; i < LightInfoBuffer.spot_light_count; i++)
    {
        SpotLight light = SpotLightBuffer.Load(i);
        LightLiSample light_sample = light.SampleLi(li_ctx, 0.f);
        float3 f = material.bsdf.Eval(interaction.isect.wo, light_sample.wi, TransportMode_Radiance);
        float3 color = f * light_sample.L;
        if (light.cast_shadow)
        {
            color *= CalculateSpotLightShadow(light, light.shadow_id, interaction);
            spot_light_id++;
        }
        radiance += color;
    }
#endif
    
    // Handle directional light DI
#ifdef HAS_DIRECTIONAL_LIGHT
    uint directional_light_id = 0;
    for (uint i = 0; i < LightInfoBuffer.directional_light_count; i++)
    {
        DirectionalLight light = DirectionalLightBuffer.Load(i);
        LightLiSample light_sample = light.SampleLi(li_ctx, 0.f);
        float3 f = material.bsdf.Eval(interaction.isect.wo, light_sample.wi, TransportMode_Radiance);
        float3 color = f * light_sample.L;
        if (light.cast_shadow)
        {
            color *= CalculateDirectionalLightShadow(light, light.shadow_id, interaction, linear_z);
            directional_light_id++;
        }
        radiance += color;
    }
#endif
    
#ifdef HAS_ENV_LIGHT
    {
#ifdef HAS_IRRADIANCE_SH
        float3 F0 = float3(0.0, 0.0, 0.0);
        F0 = lerp(F0, g_buffer_data.albedo.rgb, g_buffer_data.metallic);
        float3 F = F0 + (max(float3(1.0 - g_buffer_data.roughness, 1.0 - g_buffer_data.roughness, 1.0 - g_buffer_data.roughness), F0) * pow(1.0 - max(dot(g_buffer_data.normal, interaction.isect.wo), 0.0), 5.0));
        float3 Kd = (1.0 - F) * (1.0 - g_buffer_data.metallic);

        float3 irradiance = float3(0.0, 0.0, 0.0);
        SH9 basis = EvaluateSH(g_buffer_data.normal);
        for (uint i = 0; i < 9; i++)
        {
            irradiance += IrradianceSH[uint2(i, 0)].rgb * basis.weights[i];
        }
        irradiance = max(float3(0.0, 0.0, 0.0), irradiance) * InvPI;
        
        float3 diffuse = irradiance * g_buffer_data.albedo.rgb;
        ambient += Kd * diffuse;
#endif

#ifdef  HAS_PREFILTER_MAP
        float3 prefiltered_color = PrefilterMap.SampleLevel(PrefilterMapSampler, reflect(-interaction.isect.wo, g_buffer_data.normal), g_buffer_data.roughness * PREFILTER_MIP_LEVELS).rgb;
        float2 brdf = GGXPreintegration.SampleLevel(PrefilterMapSampler, float2(clamp(dot(g_buffer_data.normal, interaction.isect.wo), 0.0, 1.0), g_buffer_data.roughness), 0.0).rg;
        float3 specular = prefiltered_color * (F * brdf.x + brdf.y);
        ambient += specular;
#endif
    }
#endif
    
    EnvDirectIllumination[pixel] = float4(ambient, 1.f);
    LightDirectIllumination[pixel] = float4(radiance, 1.f);
    PositionDepth[pixel] = float4(interaction.isect.p, linear_z);
    NormalRoughness[pixel] = float4(g_buffer_data.normal * 0.5f + 0.5f, g_buffer_data.roughness);
    AlbedoMetallic[pixel] = float4(g_buffer_data.albedo, g_buffer_data.metallic);
}