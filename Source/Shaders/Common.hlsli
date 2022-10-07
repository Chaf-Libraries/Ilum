#ifndef COMMON_HLSLI
#define COMMON_HLSLI

struct ViewInfo
{
    float4x4 view_matrix;
    float4x4 projection_matrix;
    float4x4 view_projection_matrix;
    float3 position;
    uint frame_count;
};

struct Vertex
{
    float3 position;
    float3 normal;
    float3 tangent;
    float2 texcoord;
};

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint3 GroupThreadID : SV_GroupThreadID;
    uint3 GroupID : SV_GroupID;
};

struct MeshletBound
{
    float3 center;
    float radius;
    float3 cone_axis;
    float cone_cut_off;
};

struct InstanceData
{
    float4x4 transform;

    float3 aabb_min;
    uint material_id;

    float3 aabb_max;
    uint model_id;
    
    uint meshlet_count;
    uint meshlet_offset;
    uint submesh_id;
};

struct Meshlet
{
    MeshletBound bound;
    uint indices_offset;
    uint indices_count;
    uint vertices_offset;
    uint vertices_count;
    uint meshlet_vertices_offset;
    uint meshlet_primitive_offset;
};

void UnPackTriangle(uint encode, out uint v0, out uint v1, out uint v2)
{
    v0 = encode & 0xff;
    v1 = (encode >> 8) & 0xff;
    v2 = (encode >> 16) & 0xff;
}

uint PackVisibilityBuffer(uint instance_id, uint meshlet_id, uint primitive_id)
{
	// Primitive ID 7
	// Meshlet ID 14
	// Instance ID 11
    uint vbuffer = 0;
    vbuffer += primitive_id & 0x7f;
    vbuffer += (meshlet_id & 0x3fff) << 7;
    vbuffer += (instance_id & 0x3ff) << 21;
    return vbuffer;
}

void UnPackVisibilityBuffer(uint visibility_buffer, out uint instance_id, out uint meshlet_id, out uint primitive_id)
{
	// Primitive ID 7
	// Meshlet ID 14
	// Instance ID 11
    primitive_id = visibility_buffer & 0x7f;
    meshlet_id = (visibility_buffer >> 7) & 0x3fff;
    instance_id = (visibility_buffer >> 21) & 0x3ff;
}

#endif