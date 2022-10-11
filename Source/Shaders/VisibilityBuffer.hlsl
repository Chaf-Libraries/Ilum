#include "Common.hlsli"

ConstantBuffer<ViewInfo> View;
StructuredBuffer<Vertex> VertexBuffer[];
StructuredBuffer<uint> IndexBuffer[];
StructuredBuffer<uint> MeshletVertexBuffer[];
StructuredBuffer<uint> MeshletPrimitiveBuffer[];
StructuredBuffer<Meshlet> MeshletBuffer[];
StructuredBuffer<InstanceData> InstanceBuffer;
Texture2D<float> TextureArray[];

struct VertexOut
{
    float4 Position : SV_Position;
    float2 Texcoord : TEXCOORD0;
};

struct PrimitiveOut
{
    uint InstanceID : COLOR1;
    uint PrimitiveID : COLOR2;
    uint MeshletID : COLOR3;
};

struct PayLoad
{
    uint meshletIndices[32];
};

[[vk::push_constant]]
struct
{
    uint instance_idx;
} push_constants;

groupshared PayLoad s_payload;

[numthreads(32, 1, 1)]
void ASmain(CSParam param)
{
    bool visible = false;
    uint meshlet_count = 0;
    uint stride = 0;
    
    InstanceData instance = InstanceBuffer[push_constants.instance_idx];
    uint model_id = instance.model_id;
    uint meshlet_id = param.DispatchThreadID.x;
    
    if (meshlet_id < instance.meshlet_count)
    {
        visible = true;
    }
    
    if (visible)
    {
        uint index = WavePrefixCountBits(visible);
        s_payload.meshletIndices[index] = meshlet_id;
    }
    
    uint visible_count = WaveActiveCountBits(visible);

    DispatchMesh(visible_count, 1, 1, s_payload);
}

[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void MSmain(CSParam param, in payload PayLoad pay_load, out vertices VertexOut verts[64], out indices uint3 tris[124], out primitives PrimitiveOut prims[124])
{
    InstanceData instance = InstanceBuffer[push_constants.instance_idx];
    uint model_id = instance.model_id;
    
    uint meshlet_id = pay_load.meshletIndices[param.GroupID.x] + instance.meshlet_offset;
   
    if (meshlet_id >= instance.meshlet_count + instance.meshlet_offset)
    {
        return;
    }
    
    float4x4 transform = instance.transform;
    
    Meshlet meshlet = MeshletBuffer[model_id][meshlet_id];
    SetMeshOutputCounts(meshlet.vertices_count, meshlet.indices_count / 3);
    
    for (uint i = param.GroupThreadID.x; i < meshlet.vertices_count; i += 32)
    {
        uint vertex_index = meshlet.vertices_offset + MeshletVertexBuffer[model_id][i + meshlet.meshlet_vertices_offset];
        Vertex vertex = VertexBuffer[model_id][vertex_index];
        
        verts[i].Position = mul(View.view_projection_matrix, mul(transform, float4(vertex.position.xyz, 1.0)));
        verts[i].Texcoord = vertex.texcoord.xy;
    }
    
    for (i = param.GroupThreadID.x; i < meshlet.indices_count / 3; i += 32)
    {
        prims[i].InstanceID = push_constants.instance_idx;
        prims[i].MeshletID = meshlet_id;
        prims[i].PrimitiveID = i;
       
        uint v0, v1, v2;
        UnPackTriangle(MeshletPrimitiveBuffer[model_id][i + meshlet.meshlet_primitive_offset], v0, v1, v2);
        tris[i] = uint3(v0, v1, v2);
    }
}

uint PSmain(VertexOut vert, PrimitiveOut prim) : SV_TARGET0
{
    return PackVisibilityBuffer(prim.InstanceID, prim.MeshletID, prim.PrimitiveID);
}