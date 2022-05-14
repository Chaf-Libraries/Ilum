#ifndef __COMMON_HLSL__
#define __COMMON_HLSL__

struct Interaction
{
    float3 position;
    float3 normal;
    float3 ffnormal;
    float3 tangent;
    float3 bitangent;
    float2 texCoord;
    float3 wo;
    
    void CreateCoordinateSystem()
    {
        const float3 ref = abs(dot(ffnormal, float3(0, 1, 0))) > 0.99f ? float3(0, 0, 1) : float3(0, 1, 0);

        tangent = normalize(cross(ref, ffnormal));
        bitangent = cross(ffnormal, tangent);
    }
    
    float3 WorldToLocal(float3 w)
    {
        return float3(dot(w, tangent), dot(w, bitangent), dot(w, normal));
    }
    
    float3 LocalToWorld(float3 w)
    {
        return tangent * w.x + bitangent * w.y + normal * w.z;
    }
    
    float3 WorldToLocalDir(float3 w)
    {
        return float3(dot(w, tangent), dot(w, bitangent), dot(w, ffnormal));
    }
    
    float3 LocalToWorldDir(float3 w)
    {
        return tangent * w.x + bitangent * w.y + ffnormal * w.z;
    }
    
    bool IsSurfaceInteraction()
    {
        return ffnormal.x != 0.0 || ffnormal.y != 0.0 || ffnormal.z != 0.0;
    }
};

float2 OctWrap(float2 v)
{
    return float2((1.0f - abs(v.y)) * (v.x >= 0.0f ? 1.0f : -1.0f), (1.0f - abs(v.x)) * (v.y >= 0.0f ? 1.0f : -1.0f));
}

float2 PackNormal(float3 n)
{
    float2 p = float2(n.x, n.y) * (1.0f / (abs(n.x) + abs(n.y) + abs(n.z)));
    p = (n.z < 0.0f) ? OctWrap(p) : p;
    return p;
}

float3 UnPackNormal(float2 p)
{
    float3 n = float3(p.x, p.y, 1.0f - abs(p.x) - abs(p.y));
    float2 tmp = (n.z < 0.0f) ? OctWrap(float2(n.x, n.y)) : float2(n.x, n.y);
    n.x = tmp.x;
    n.y = tmp.y;
    return normalize(n);
}

static const uint MetalRoughnessWorkflow = 0;
static const uint SpecularGlossinessWorkflow = 1;
#define MAX_TEXTURE_ARRAY_SIZE 1024
struct MaterialInfo
{
    float3 albedo;
    float metallic;
    float roughness;
    float ior;
    float3 emissive;
    float sheen;
    float sheen_tint;
    float clearcoat;
    float clearcoat_roughness;
    float specular;
    float specular_tint;
    float3 specular_color;
    float subsurface;
    float3 transmission;
    float occlusion;
    float ax;
    float ay;
    bool thin;
};

struct ShadingState
{
    uint trace_depth;
    float eta;
    
    float3 position;
    float2 uv;
    float3 normal;
    float3 ffnormal;
    float3 tangent;
    float3 bitangent;
    float depth;
    
    float2 dx;
    float2 dy;
    float3 bary;
    
    bool is_subsurface;
    
    uint matID;
    MaterialInfo mat;
};
#endif