#include "Common.hlsli"

StructuredBuffer<InstanceData> InstanceBuffer;
StructuredBuffer<Meshlet> MeshletBuffer[];
StructuredBuffer<uint> MeshletDataBuffer[];
StructuredBuffer<Vertex> VertexBuffer[];
StructuredBuffer<uint> IndexBuffer[];
ConstantBuffer<View> ViewBuffer;

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
    uint MeshletIndices[32];
    uint InstanceIndices[32];
};

groupshared PayLoad s_payload;

[numthreads(8, 8, 1)]
void ASmain(CSParam param)
{
    bool visible = false;
    
    uint instance_id = param.DispatchThreadID.x;
    uint meshlet_id = param.DispatchThreadID.y;
    
    uint instance_count = 0;
    uint instance_stride = 0;
    InstanceBuffer.GetDimensions(instance_count, instance_stride);
    
    if (instance_count <= instance_id)
    {
        return;
    }
    
    InstanceData instance = InstanceBuffer[instance_id];
    
    uint meshlet_count = 0;
    uint meshlet_stride = 0;
    MeshletBuffer[instance.mesh_id].GetDimensions(meshlet_count, meshlet_stride);
    
    if (meshlet_count <= meshlet_id)
    {
        return;
    }
    
    Meshlet meshlet = MeshletBuffer[instance.mesh_id][meshlet_id];
    
    visible = true;
    
    if (visible)
    {
        uint index = WavePrefixCountBits(visible);
        s_payload.InstanceIndices[index] = instance_id;
        s_payload.MeshletIndices[index] = meshlet_id;
    }
    
    uint visible_count = WaveActiveCountBits(visible);

    DispatchMesh(visible_count, 1, 1, s_payload);
}

[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void MSmain(CSParam param, in payload PayLoad pay_load, out vertices VertexOut verts[64], out indices uint3 tris[124], out primitives PrimitiveOut prims[124])
{
    uint instance_id = pay_load.InstanceIndices[param.GroupID.x];
    uint meshlet_id = pay_load.MeshletIndices[param.GroupID.x];
    
    InstanceData instance = InstanceBuffer[instance_id];
    Meshlet meshlet = MeshletBuffer[instance.mesh_id][meshlet_id];
    
    uint meshlet_vertices_count = meshlet.GetVertexCount();
    uint meshlet_triangle_count = meshlet.GetTriangleCount();
    
    SetMeshOutputCounts(meshlet_vertices_count, meshlet_triangle_count);
    
    for (uint i = param.GroupThreadID.x; i < meshlet_vertices_count; i += 32)
    {
        uint vertex_id = MeshletDataBuffer[instance.mesh_id][meshlet.data_offset + i];
        Vertex vertex = VertexBuffer[instance.mesh_id][vertex_id];
        
        verts[i].Position = mul(ViewBuffer.view_projection_matrix, mul(instance.transform, float4(vertex.position.xyz, 1.0)));
        verts[i].Texcoord = vertex.texcoord.xy;
    }
    
    for (uint i = param.GroupThreadID.x; i < meshlet_triangle_count; i += 32)
    {
        prims[i].InstanceID = instance_id;
        prims[i].MeshletID = meshlet_id;
        prims[i].PrimitiveID = i;
       
        uint v0, v1, v2;
        UnPackTriangle(MeshletDataBuffer[instance.mesh_id][meshlet.data_offset + meshlet_vertices_count + i], v0, v1, v2);
        tris[i] = uint3(v0, v1, v2);
    }
}

float PSmain(VertexOut vert, PrimitiveOut prim) : SV_TARGET0
{
    //return PackVisibilityBuffer(prim.InstanceID, prim.MeshletID, prim.PrimitiveID);
    return 1.f;
}