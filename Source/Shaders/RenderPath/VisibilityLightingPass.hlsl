#include "../Common.hlsli"

#ifndef RUNTIME
#define HAS_MESH
#define HAS_SKINNED_MESH
#define HAS_POINT_LIGHT
#endif

Texture2D<uint> VisibilityBuffer;
Texture2D<float> DepthBuffer;
RWTexture2D<float4> Output;

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
        dispatch_id.y >= extent.y)
    {
        return;
    }
    
    if (DepthBuffer.Load(int3(dispatch_id, 0)).r > 3e38f)
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
        InterlockedAdd(MaterialCountBuffer[material_id], 1);
    }
}

RWStructuredBuffer<uint> MaterialOffsetBuffer;

groupshared uint shared_data[256];

[numthreads(128, 1, 1)]
void CalculateMaterialOffset(CSParam param)
{
    uint dispatch_id = param.DispatchThreadID.x;
    
    uint material_count, temp;
    MaterialCountBuffer.GetDimensions(material_count, temp);
    
    if (dispatch_id * 2 + 1 >= material_count)
    {
        return;
    }
    
    uint rd_id, wr_id, mask;
    const uint steps = uint(log2(dispatch_id)) + 1;
    
    shared_data[dispatch_id * 2] = MaterialCountBuffer[dispatch_id * 2];
    shared_data[dispatch_id * 2 + 1] = MaterialCountBuffer[dispatch_id * 2 + 1];
    
    GroupMemoryBarrierWithGroupSync();
    
    for (uint step = 0; step < steps; step++)
    {
        mask = (1u << step) - 1u;
        rd_id = ((dispatch_id >> step) << (step + 1u)) + mask;
        wr_id = rd_id + 1u + (dispatch_id & mask);
        shared_data[wr_id] += shared_data[rd_id];
        
        GroupMemoryBarrierWithGroupSync();
    }
    
    MaterialOffsetBuffer[dispatch_id * 2 + 1] = shared_data[dispatch_id * 2];
    MaterialOffsetBuffer[dispatch_id * 2 + 2] = shared_data[dispatch_id * 2 + 1];
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
    
    uint temp;
    InterlockedAdd(IndirectCommandBuffer[material_id].x, 1, temp);
    MaterialPixelBuffer[temp + MaterialOffsetBuffer[material_id]] = PackXY(dispatch_id.x, dispatch_id.y);
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

ConstantBuffer<View> ViewBuffer;
ConstantBuffer<LightInfo> LightInfoBuffer;

#ifdef HAS_POINT_LIGHT
StructuredBuffer<PointLight> PointLightBuffer;
#endif

#ifdef HAS_MESH
StructuredBuffer<Vertex> MeshVertexBuffer[];
StructuredBuffer<uint> MeshIndexBuffer[];
#endif

#ifdef HAS_SKINNED_MESH
StructuredBuffer<SkinnedVertex> SkinnedMeshVertexBuffer[];
StructuredBuffer<uint> SkinnedMeshIndexBuffer[];
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
    mesh_type = 0;
    SurfaceInteraction interaction;
    
    float3 bary, bary_ddx, bary_ddy;
    
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
        interaction.isect.uv = v0.texcoord0.xy * bary.x + v1.texcoord0.xy * bary.y + v2.texcoord0.xy * bary.z;
        interaction.isect.n = v0.normal.xyz * bary.x + v1.normal.xyz * bary.y + v2.normal.xyz * bary.z;
        interaction.isect.n = normalize(mul((float4x3) instance.transform, normalize(interaction.isect.n)).xyz);
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
    }
#endif
    
    Material material;
    material.Init(interaction);
    
    float3 radiance = 0.f;
    
    LightSampleContext li_ctx;
    li_ctx.n = interaction.isect.n;
    li_ctx.ns = interaction.shading_n;
    li_ctx.p = interaction.isect.p;
    
#ifdef HAS_POINT_LIGHT
    for (uint i = 0; i < LightInfoBuffer.point_light_count; i++)
    {
        PointLight light = PointLightBuffer.Load(i);
        LightLiSample light_sample = light.SampleLi(li_ctx, 0.f);
        
        float3 f = material.bsdf.Eval(interaction.isect.wo, light_sample.wi, TransportMode_Radiance) * abs(dot(light_sample.wi, interaction.shading_n));
        radiance += f * light_sample.L;
    }
#endif
    
    Output[pixel] = float4(radiance, 1.f);
    
    //Material material;
    //material.Init(interaction);
    
    //Output[pixel] = float4(material.bsdf.Eval(1, 1, TransportMode_Radiance), 1);
    //Output[pixel] = 1.f;
}