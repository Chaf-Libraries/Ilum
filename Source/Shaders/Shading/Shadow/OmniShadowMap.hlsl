#include "../../Common.hlsli"
#include "../../Light.hlsli"

StructuredBuffer<Instance> InstanceBuffer;
StructuredBuffer<Meshlet> MeshletBuffer[];
StructuredBuffer<uint> MeshletDataBuffer[];
StructuredBuffer<uint> IndexBuffer[];
StructuredBuffer<PointLight> PointLightBuffer;
ConstantBuffer<LightInfo> LightInfoBuffer;

static const float4x4 ViewProjection[6] =
{
    float4x4(
        0, 0, 1.0001999, 1,
        0, 1, 0, 0,
        1, 0, 0, 0,
        0, 0, -0.20002, 0
    ),
    float4x4(
        0, 0, -1.0001999, -1,
        0, 1, 0, 0,
        -1, 0, 0, 0,
        0, 0, -0.20002, 0
    ),
    float4x4(
        1, 0, 0, 0,
        0, 0, 1.0001999, 1,
        0, 1, 0, 0,
        0, 0, -0.20002, 0
    ),
    float4x4(
        1, 0, 0, 0,
        0, 0, -1.0001999, -1,
        0, -1, 0, 0,
        0, 0, -0.20002, 0
    ),
    float4x4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, -1.0001999, -1,
        0, 0, -0.20002, 0
    ),
    float4x4(
        -1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1.0001999, 1,
        0, 0, -0.20002, 0
    ),
};

#ifdef HAS_SKINNED
StructuredBuffer<SkinnedVertex> VertexBuffer[];
StructuredBuffer<float4x4> BoneMatrices[];

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
#else
StructuredBuffer<Vertex> VertexBuffer[];

struct VertexIn
{
    float3 Position : POSITION0;
    float3 Normal : NORMAL0;
    float3 Tangent : TANGENT0;
    float2 Texcoord0 : TEXCOORD0;
    float2 Texcoord1 : TEXCOORD1;
    uint InstanceID : SV_InstanceID;
};
#endif

struct VertexOut
{
    float4 Position : SV_Position;
    uint InstanceID : COLOR0;
};

struct PrimitiveOut
{
    uint Layer : SV_RenderTargetArrayIndex;
};

struct PayLoad
{
    uint MeshletIndices[32];
    uint InstanceIndices[32];
    uint LayerIndices[32];
};

groupshared PayLoad s_payload;

[numthreads(32, 1, 1)]
void ASmain(CSParam param)
{
    bool visible = false;
    
    uint meshlet_id = param.DispatchThreadID.x;
    uint instance_id = param.DispatchThreadID.y;
    uint layer_id = param.DispatchThreadID.z;
    
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
#ifdef HAS_SKINNED
            visible = true;
#else
           // Meshlet meshlet = MeshletBuffer[instance.mesh_id][meshlet_id];
           // visible = ViewBuffer.IsInsideFrustum(meshlet, instance.transform);
            visible = true;
#endif
        }
    }

    if (visible)
    {
        uint index = WavePrefixCountBits(visible);
        s_payload.InstanceIndices[index] = instance_id;
        s_payload.MeshletIndices[index] = meshlet_id;
        s_payload.LayerIndices[index] = layer_id;
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
    uint layer_id = pay_load.LayerIndices[param.GroupID.x];
    uint light_id = layer_id / 6;
    
    Instance instance = InstanceBuffer[instance_id];
    Meshlet meshlet = MeshletBuffer[instance.mesh_id][meshlet_id];
    PointLight light = PointLightBuffer[light_id];
    
    uint meshlet_vertices_count = meshlet.vertex_count;
    uint meshlet_triangle_count = meshlet.triangle_count;
    
    SetMeshOutputCounts(meshlet_vertices_count, meshlet_triangle_count);
    
    for (uint i = param.GroupThreadID.x; i < meshlet_vertices_count; i += 32)
    {
        uint vertex_id = MeshletDataBuffer[instance.mesh_id][meshlet.data_offset + i];
        
#ifdef HAS_SKINNED
        SkinnedVertex vertex = VertexBuffer[instance.mesh_id][vertex_id];
        
        if (instance.animation_id != ~0U)
        {
            uint bone_count = 0;
            uint bone_stride = 0;
            BoneMatrices[instance.animation_id].GetDimensions(bone_count, bone_stride);
            
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
            verts[i].Position = mul(transpose(ViewProjection[layer_id % 6]), float4(mul(instance.transform, float4(total_position.xyz, 1.0)).xyz - light.position, 1.f));
        }
        else
        {
            verts[i].Position = mul(transpose(ViewProjection[layer_id % 6]), float4(mul(instance.transform, float4(vertex.position.xyz, 1.0)).xyz - light.position, 1.f));
        }
#else
        Vertex vertex = VertexBuffer[instance.mesh_id][vertex_id];
        verts[i].Position = mul(transpose(ViewProjection[layer_id % 6]), float4(mul(instance.transform, float4(vertex.position.xyz, 1.0)).xyz - light.position, 1.f));
#endif
    }
    
    for (uint i = param.GroupThreadID.x; i < meshlet_triangle_count; i += 32)
    {
        uint v0, v1, v2;
        UnPackTriangle(MeshletDataBuffer[instance.mesh_id][meshlet.data_offset + meshlet.vertex_count + meshlet.triangle_count + i], v0, v1, v2);
        
        tris[i] = uint3(v0, v1, v2);
        prims[i].Layer = layer_id;
    }
}

void PSmain(VertexOut vert)
{
}

void VSmain(in VertexIn vert_in, out VertexOut vert_out)
{
    Instance instance = InstanceBuffer[vert_in.InstanceID];
    
#ifdef HAS_SKINNED
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
        //vert_out.Position = mul(ViewBuffer.view_projection, mul(instance.transform, float4(total_position.xyz, 1.0)));
    }
    else
    {
       // vert_out.Position = mul(ViewBuffer.view_projection, mul(instance.transform, float4(vert_in.Position.xyz, 1.0)));
    }
#else
    //vert_out.Position = mul(ViewBuffer.view_projection_matrix, mul(instance.transform, float4(vert_in.Position.xyz, 1.0)));
#endif
}

void FSmain(VertexOut vert, uint PrimitiveID : SV_PrimitiveID)
{
}