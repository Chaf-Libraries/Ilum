#include "Common.hlsli"

StructuredBuffer<InstanceData> InstanceBuffer;
StructuredBuffer<Meshlet> MeshletBuffer[];
StructuredBuffer<uint> MeshletDataBuffer[];
StructuredBuffer<Vertex> VertexBuffer[];
StructuredBuffer<uint> IndexBuffer[];
ConstantBuffer<View> ViewBuffer;

struct VertexIn
{
    float3 Position : POSITION0;
    float3 Normal : NORMAL0;
    float3 Tangent : TANGENT0;
    float2 Texcoord0 : TEXCOORD0;
    float2 Texcoord1 : TEXCOORD1;
    uint InstanceID : SV_InstanceID;
};

struct VertexOut
{
    float4 Position : SV_Position;
    float2 Texcoord : TEXCOORD0;
};

struct PrimitiveOut
{
    uint InstanceID : COLOR1;
    uint PrimitiveID : COLOR2;
};

struct PayLoad
{
    uint MeshletIndices[32];
    uint InstanceIndices[32];
};

groupshared PayLoad s_payload;

[numthreads(32, 1, 1)]
void ASmain(CSParam param)
{
    bool visible = false;
    
    uint instance_id = param.DispatchThreadID.y;
    uint meshlet_id = param.DispatchThreadID.x;
    
    uint instance_count = 0;
    uint instance_stride = 0;
    InstanceBuffer.GetDimensions(instance_count, instance_stride);
    
    if (instance_id < instance_count)
    {
        InstanceData instance = InstanceBuffer[instance_id];
        
        uint meshlet_count = 0;
        uint meshlet_stride = 0;
        MeshletBuffer[instance.mesh_id].GetDimensions(meshlet_count, meshlet_stride);
        
        if (meshlet_id < meshlet_count)
        {
            visible = true;
        }
    }

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
    
    uint meshlet_vertices_count = meshlet.vertex_count;
    uint meshlet_triangle_count = meshlet.triangle_count;
    
    SetMeshOutputCounts(meshlet_vertices_count, meshlet_triangle_count);
    
    for (uint i = param.GroupThreadID.x; i < meshlet_vertices_count; i += 32)
    {
        uint vertex_id = MeshletDataBuffer[instance.mesh_id][meshlet.data_offset + i];
        Vertex vertex = VertexBuffer[instance.mesh_id][vertex_id];
        
        verts[i].Position = mul(ViewBuffer.view_projection_matrix, mul(instance.transform, float4(vertex.position.xyz, 1.0)));
        verts[i].Texcoord = vertex.texcoord0.xy;
    }
    
    for (uint i = param.GroupThreadID.x; i < meshlet_triangle_count; i += 32)
    {
        uint primitive_id = MeshletDataBuffer[instance.mesh_id][meshlet.data_offset + meshlet_vertices_count + i];
        
        prims[i].InstanceID = instance_id;
        prims[i].PrimitiveID = primitive_id;
        
        uint v0 = IndexBuffer[instance.mesh_id][primitive_id * 3] - meshlet.vertex_offset;
        uint v1 = IndexBuffer[instance.mesh_id][primitive_id * 3 + 1] - meshlet.vertex_offset;
        uint v2 = IndexBuffer[instance.mesh_id][primitive_id * 3 + 2] - meshlet.vertex_offset;
        
        tris[i] = uint3(v0, v1, v2);
    }
}

uint PSmain(VertexOut vert, PrimitiveOut prim) : SV_TARGET0
{
    return PackVisibilityBuffer(prim.InstanceID, prim.PrimitiveID);
}

void VSmain(in VertexIn vert_in, out VertexOut vert_out, out PrimitiveOut prim)
{
    InstanceData instance = InstanceBuffer[vert_in.InstanceID];
    vert_out.Position = mul(ViewBuffer.view_projection_matrix, mul(instance.transform, float4(vert_in.Position.xyz, 1.0)));
    vert_out.Texcoord = vert_in.Texcoord0.xy;
}

uint FSmain(VertexOut vert, PrimitiveOut prim, uint PrimitiveID : SV_PrimitiveID) : SV_Target0
{
    return PackVisibilityBuffer(prim.InstanceID, PrimitiveID);
}