#include "Common.hlsli"

StructuredBuffer<Instance> InstanceBuffer;
StructuredBuffer<Meshlet> MeshletBuffer[];
StructuredBuffer<uint> MeshletDataBuffer[];
StructuredBuffer<SkinnedVertex> VertexBuffer[];
StructuredBuffer<uint> IndexBuffer[];
StructuredBuffer<float4x4> BoneMatrices[];
ConstantBuffer<View> ViewBuffer;
RWStructuredBuffer<float> DebugBuffer;

struct VertexIn
{
    float3 Position : POSITION0;
    float3 Normal : NORMAL0;
    float3 Tangent : TANGENT0;
    float2 Texcoord0 : TEXCOORD0;
    float2 Texcoord1 : TEXCOORD1;
    int4 Bones0 : BLENDINDICES0;
    int4 Bones1 : BLENDINDICES1;
    float4 Weights0 : BLENDWEIGHT0;
    float4 Weights1 : BLENDWEIGHT1;
    uint InstanceID : SV_InstanceID;
};

struct VertexOut
{
    float4 Position : SV_Position;
    float2 Texcoord : TEXCOORD0;
    uint InstanceID : COLOR0;
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
        Instance instance = InstanceBuffer[instance_id];
        
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
    
    Instance instance = InstanceBuffer[instance_id];
    Meshlet meshlet = MeshletBuffer[instance.mesh_id][meshlet_id];
    
    uint meshlet_vertices_count = meshlet.vertex_count;
    uint meshlet_triangle_count = meshlet.triangle_count;
    
    SetMeshOutputCounts(meshlet_vertices_count, meshlet_triangle_count);
    
    for (uint i = param.GroupThreadID.x; i < meshlet_vertices_count; i += 32)
    {
        uint vertex_id = MeshletDataBuffer[instance.mesh_id][meshlet.data_offset + i];
        SkinnedVertex vertex = VertexBuffer[instance.mesh_id][vertex_id];
        
        if (instance.animation_id != ~0U)
        {
            uint bone_count = 0;
            uint bone_stride = 0;
            BoneMatrices[instance.animation_id].GetDimensions(bone_count, bone_stride);
            
            DebugBuffer[0] = VertexBuffer[0][0].position[0];
            DebugBuffer[1] = VertexBuffer[0][0].position[1];
            DebugBuffer[2] = VertexBuffer[0][0].position[2];
            DebugBuffer[3] = VertexBuffer[0][0].tangent[0];
            DebugBuffer[4] = VertexBuffer[0][0].tangent[1];
            DebugBuffer[5] = VertexBuffer[0][0].tangent[2];
            DebugBuffer[6] = VertexBuffer[0][0].texcoord0[0];
            DebugBuffer[7] = VertexBuffer[0][0].texcoord0[1];
            DebugBuffer[8] = VertexBuffer[0][0].texcoord1[0];
            DebugBuffer[9] = VertexBuffer[0][0].texcoord1[1];
            DebugBuffer[10] = VertexBuffer[0][0].bones[0];
            DebugBuffer[11] = VertexBuffer[0][0].bones[1];
            DebugBuffer[12] = VertexBuffer[0][0].bones[2];
            DebugBuffer[13] = VertexBuffer[0][0].bones[0];
            DebugBuffer[14] = VertexBuffer[0][0].weights[0];
            DebugBuffer[15] = VertexBuffer[0][0].weights[1];
            DebugBuffer[16] = VertexBuffer[0][0].weights[2];
            DebugBuffer[17] = VertexBuffer[0][0].weights[0];
            
            float4 total_position = 0.f;
            for (uint i = 0; i < MAX_BONE_INFLUENCE; i++)
            {
                if (vertex.bones[i] == -1)
                {
                    continue;
                }
                if (vertex.bones[i] >= bone_count)
                {
                    total_position = float4(vertex.position, 1.0f);
                    break;
                }
                float4 local_position = mul(BoneMatrices[instance.animation_id][vertex.bones[i]], float4(vertex.position, 1.0f));
                total_position += local_position * vertex.weights[i];
            }
            verts[i].Position = mul(ViewBuffer.view_projection_matrix, mul(instance.transform, float4(total_position.xyz, 1.0)));
        }
        else
        {
            verts[i].Position = mul(ViewBuffer.view_projection_matrix, mul(instance.transform, float4(vertex.position.xyz, 1.0)));
        }
        
        verts[i].Texcoord = vertex.texcoord0.xy;
    }
    
    for (uint i = param.GroupThreadID.x; i < meshlet_triangle_count; i += 32)
    {
        uint primitive_id = MeshletDataBuffer[instance.mesh_id][meshlet.data_offset + meshlet.vertex_count + i];
        
        prims[i].InstanceID = instance_id;
        prims[i].PrimitiveID = primitive_id;
        
        uint v0, v1, v2;
        UnPackTriangle(MeshletDataBuffer[instance.mesh_id][meshlet.data_offset + meshlet.vertex_count + meshlet.triangle_count + i], v0, v1, v2);
        
        tris[i] = uint3(v0, v1, v2);
    }
}

uint PSmain(VertexOut vert, PrimitiveOut prim) : SV_TARGET0
{
    return PackVisibilityBuffer(prim.InstanceID, prim.PrimitiveID);
}

void VSmain(in VertexIn vert_in, out VertexOut vert_out, out PrimitiveOut prim)
{
    Instance instance = InstanceBuffer[vert_in.InstanceID];
    if (instance.animation_id != ~0U)
    {
        uint bone_count = 0;
        uint bone_stride = 0;
        BoneMatrices[instance.animation_id].GetDimensions(bone_count, bone_stride);
            
        float4 total_position = 0.f;
        for (uint i = 0; i < MAX_BONE_INFLUENCE; i++)
        {
            int bone = -1;
            float weight = 0.f;
            
            if (i < 4)
            {
                bone = vert_in.Bones0[i];
                weight = vert_in.Weights0[i];
            }
            else
            {
                bone = vert_in.Bones1[i - 4];
                weight = vert_in.Weights1[i - 4];
            }
            
            if (bone == -1)
            {
                continue;
            }
            if (bone >= bone_count)
            {
                total_position = float4(vert_in.Position, 1.0f);
                break;
            }

            float4 local_position = mul(BoneMatrices[instance.animation_id][bone], float4(vert_in.Position, 1.0f));
            total_position += local_position * weight;
        }
        vert_out.Position = mul(ViewBuffer.view_projection_matrix, mul(instance.transform, float4(total_position.xyz, 1.0)));
    }
    else
    {
        vert_out.Position = mul(ViewBuffer.view_projection_matrix, mul(instance.transform, float4(vert_in.Position.xyz, 1.0)));
    }
    vert_out.Texcoord = vert_in.Texcoord0.xy;
    vert_out.InstanceID = vert_in.InstanceID;

}

uint FSmain(VertexOut vert, uint PrimitiveID : SV_PrimitiveID) : SV_Target0
{
    return PackVisibilityBuffer(vert.InstanceID, PrimitiveID);
}