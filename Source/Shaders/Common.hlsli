#ifndef COMMON_HLSLI
#define COMMON_HLSLI

#include "Math.hlsli"
#include "RayTracingUtlis.hlsli"
#include "LightSource.hlsli"

struct ViewInfo
{
    float4x4 view_matrix;
    float4x4 inv_view_matrix;
    float4x4 projection_matrix;
    float4x4 inv_projection_matrix;
    float4x4 view_projection_matrix;
    float3 position;
    uint frame_count;
    
    RayDesc CastRay(float2 sceneUV)
    {
        RayDesc ray;

        float4 target = mul(inv_projection_matrix, float4(sceneUV.x, sceneUV.y, 1, 1));

        ray.Origin = mul(inv_view_matrix, float4(0, 0, 0, 1)).xyz;
        ray.Direction = mul(inv_view_matrix, float4(normalize(target.xyz), 0)).xyz;
        ray.TMin = 0.0;
        ray.TMax = Infinity;

        return ray;
    }
};

struct View
{
    float4x4 view_matrix;
    float4x4 inv_view_matrix;
    float4x4 projection_matrix;
    float4x4 inv_projection_matrix;
    float4x4 view_projection_matrix;
    float3 position;
    uint frame_count;
    
    RayDesc CastRay(float2 sceneUV)
    {
        RayDesc ray;

        float4 target = mul(inv_projection_matrix, float4(sceneUV.x, sceneUV.y, 1, 1));

        ray.Origin = mul(inv_view_matrix, float4(0, 0, 0, 1)).xyz;
        ray.Direction = mul(inv_view_matrix, float4(normalize(target.xyz), 0)).xyz;
        ray.TMin = 0.0;
        ray.TMax = Infinity;

        return ray;
    }
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

//struct InstanceData
//{
//    float4x4 transform;
//
//    float3 aabb_min;
//    uint material_id;
//
//    float3 aabb_max;
//    uint model_id;
//    
//    uint meshlet_count;
//    uint meshlet_offset;
//    uint vertex_offset;
//    uint index_offset;
//};

struct InstanceData
{
    float4x4 transform;
    uint mesh_id;
    uint material_id;
};

//struct Meshlet
//{
//    MeshletBound bound;
//    uint indices_offset;
//    uint indices_count;
//    uint vertices_offset;
//    uint vertices_count;
//    uint meshlet_vertices_offset;
//    uint meshlet_primitive_offset;
//};

struct Meshlet
{
    float3 center;
    float radius;
    int cone;
    // int8_t cone_axis[3];
    // int8_t cone_cutoff;

    uint data_offset;
    uint primitive_count;
    // uint8_t vertex_count;
    // uint8_t triangle_count;
    
    uint GetVertexCount()
    {
        return primitive_count & 0xff;
    }
    
    uint GetTriangleCount()
    {
        return (primitive_count >> 8) & 0xff;
    }
};

typedef uint TransportMode;
static const TransportMode TransportMode_Radiance = 0;
static const TransportMode TransportMode_Importance = 1;

struct BSDFContext
{
    TransportMode mode;
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

//ConstantBuffer<ViewInfo> View;


//StructuredBuffer<uint> IndexBuffer[];
//StructuredBuffer<uint> MeshletVertexBuffer[];
//StructuredBuffer<uint> MeshletPrimitiveBuffer[];
//StructuredBuffer<Meshlet> MeshletBuffer[];
//StructuredBuffer<InstanceData> InstanceBuffer;
//
//Texture2D TextureArray[];
//SamplerState SamplerStateArray[];

struct Frame
{
    float3 s, t;
    float3 n;
    
    void CreateCoordinateSystem(float3 n_)
    {
        n = n_;
        if (n.z < 0.f)
        {
            const float a = 1.0f / (1.0f - n.z);
            const float b = n.x * n.y * a;
            s = float3(1.0f - n.x * n.x * a, -b, n.x);
            t = float3(b, n.y * n.y * a - 1.0f, -n.y);
        }
        else
        {
            const float a = 1.0f / (1.0f + n.z);
            const float b = -n.x * n.y * a;
            s = float3(1.0f - n.x * n.x * a, b, -n.x);
            t = float3(b, 1.0f - n.y * n.y * a, -n.y);
        }
    }
    
    float3 ToLocal(float3 v)
    {
        return float3(dot(v, s), dot(v, t), dot(v, n));
    }
    
    float3 ToWorld(float3 v)
    {
        return n * v.z + (t * v.y + s * v.x);
    }
    
    float CosTheta2(float3 v)
    {
        return v.z * v.z;
    }
    
    float CosTheta(float3 v)
    {
        return v.z;
    }
    
    float SinTheta2(float3 v)
    {
        return v.x * v.x + v.y * v.z;
    }
    
    float SinTheta(float3 v)
    {
        return sqrt(max(SinTheta2(v), 0));
    }
};

struct SurfaceInteraction
{
    // Distance traveled along the ray
    float t;
    
    // Position of the intersection in world coordinates
    float3 p;
    
    // Geometric normal
    float3 n;
    
    // UV surface coordinates
    float2 uv;
    
    // Position partials wrt. the UV parameterization
    float3 dpdu, dpdv;
    
    // Normal partials wrt. the UV parameterization
    float3 dndu, dndv;
    
    // UV partials wrt. the UV parameterization
    float3 duvdx, duvdy;
    
    // Incident direction in the local shading frame
    float3 wi;
    
    // Shading frame
    Frame sh_frame;
    
    // Material ID
    uint material_id;
    
    float3 ToWorld(float3 v)
    {
        return sh_frame.ToWorld(v);
    }
    
    float3 ToLocal(float3 v)
    {
        return sh_frame.ToLocal(v);
    }
    
//    void Init(
//        RayDesc ray, 
//        BuiltInTriangleIntersectionAttributes attributes, 
//        StructuredBuffer<Vertex> VertexBuffer[],
//        StructuredBuffer<InstanceData> InstanceBuffer,
//        StructuredBuffer<uint> IndexBuffer[])
//    {
//        const uint instance_id = InstanceIndex();
//        const uint primitive_id = PrimitiveIndex();
//        const float3 bary = float3(1.0 - attributes.barycentrics.x - attributes.barycentrics.y, attributes.barycentrics.x, attributes.barycentrics.y);
//        
//        InstanceData instance = InstanceBuffer[instance_id];
//        
//        // Vertex attribute of the triangle
//        Vertex v0 = VertexBuffer[instance.model_id][instance.vertex_offset + IndexBuffer[instance.model_id][instance.index_offset + 3 * primitive_id]];
//        Vertex v1 = VertexBuffer[instance.model_id][instance.vertex_offset + IndexBuffer[instance.model_id][instance.index_offset + 3 * primitive_id + 1]];
//        Vertex v2 = VertexBuffer[instance.model_id][instance.vertex_offset + IndexBuffer[instance.model_id][instance.index_offset + 3 * primitive_id + 2]];
//        
//        // World position
//        const float3 position = v0.position * bary.x + v1.position * bary.y + v2.position * bary.z;
//        const float3 world_position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
//        
//        // Texcoord
//        const float2 texcoord = v0.texcoord.xy * bary.x + v1.texcoord.xy * bary.y + v2.texcoord.xy * bary.z;
//        
//        // Normal
//        float3 normal = normalize(v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z);
//        const float3 world_normal = normalize(mul(WorldToObject4x3(), normal).xyz);
//         float3 geo_normal = normalize(mul(WorldToObject4x3(), normalize(cross(v2.position.xyz - v0.position.xyz, v1.position.xyz - v0.position.xyz))).xyz);
//        
//        // Interaction attribute
//        t = RayTCurrent();
//        p = world_position;
//        n = geo_normal;
//        uv = texcoord;
//        sh_frame.CreateCoordinateSystem(normal);
//        material_id = instance.material_id;
//    }
};



#endif