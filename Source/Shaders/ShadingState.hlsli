#ifndef __SHADINGSTATE_HLSL__
#define __SHADINGSTATE_HLSL__

#include "Common.hlsli"
#include "Math.hlsli"

static const uint MetalRoughnessWorkflow = 0;
static const uint SpecularGlossinessWorkflow = 1;

#define MAX_TEXTURE_ARRAY_SIZE 1024

struct MaterialInfo
{
    float3 albedo;
    float roughness;
    float metallic;
    
    
    float ior;
    
    float3 f0; // full reflectance color
    float3 f90; // reflectance color at grazing angle
};

struct ShadingState
{
    float3 position;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float3 ffnormal;
    float2 uv;
    float depth;
    
    float2 dx;
    float2 dy;
    float3 bary;
        
    uint matID;
    MaterialInfo mat_info;
    
    void LoadMaterial()
    {
        if (materials[matID].type == SpecularGlossinessWorkflow)
        {
            mat_info.f0 = materials[matID].pbr_specular_factor;
            mat_info.roughness = materials[matID].pbr_glossiness_factor;
            if (materials[matID].pbr_specular_glossiness_texture < 1024)
            {
                float4 sg_sample = texture_array[materials[matID].pbr_specular_glossiness_texture].SampleGrad(texture_sampler, uv, dx, dy);

            }
            
           
        }
    }
    
    void LoadVisibilityBuffer(Texture2D<uint> vbuffer, uint2 pixel_coord, float4x4 view_projection)
    {
        uint2 vbuffer_size;
        vbuffer.GetDimensions(vbuffer_size.x, vbuffer_size.y);
        
        uint instance_id;
        uint primitive_id;
        uint meshlet_id;
    
        UnPackVBuffer(vbuffer[pixel_coord], instance_id, meshlet_id, primitive_id);
        
        matID = instances[instance_id].material;
        uint mesh_id = instances[instance_id].mesh;
        
        float4x4 transform = instances[instance_id].transform;
        Meshlet meshlet = meshlets[mesh_id][meshlet_id];

        uint i0, i1, i2;
        UnPackTriangle(meshlet_primitives[mesh_id][primitive_id + meshlet.primitive_offset], i0, i1, i2);
        i0 = meshlet_vertices[mesh_id][meshlet.vertex_offset + i0];
        i1 = meshlet_vertices[mesh_id][meshlet.vertex_offset + i1];
        i2 = meshlet_vertices[mesh_id][meshlet.vertex_offset + i2];
        
        Vertex v0 = vertices[mesh_id][i0];
        Vertex v1 = vertices[mesh_id][i1];
        Vertex v2 = vertices[mesh_id][i2];
        
        float4 clip_pos0 = mul(view_projection, mul(transform, float4(v0.position.xyz, 1.0)));
        float4 clip_pos1 = mul(view_projection, mul(transform, float4(v1.position.xyz, 1.0)));
        float4 clip_pos2 = mul(view_projection, mul(transform, float4(v2.position.xyz, 1.0)));
        
        float2 p0 = clip_pos1.xy - clip_pos0.xy;
        float2 p1 = clip_pos0.xy - clip_pos2.xy;
        bool front = p0.x * p1.y - p1.x * p0.y;
        
        // Calculate barycentric
        float3 inv_w = rcp(float3(clip_pos0.w, clip_pos1.w, clip_pos2.w));
        
        float3 ndc0 = clip_pos0.xyz / clip_pos0.w;
        float3 ndc1 = clip_pos1.xyz / clip_pos1.w;
        float3 ndc2 = clip_pos2.xyz / clip_pos2.w;
        float2 ndc = (float2(pixel_coord) + float2(0.5, 0.5)) / float2(vbuffer_size);
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

        float3 ddx_barycentrics = (G_dx * H - G * H_dx) * (rcpH * rcpH) * (2.0 / float(vbuffer_size.x));
        float3 ddy_barycentrics = (G_dy * H - G * H_dy) * (rcpH * rcpH) * (-2.0 / float(vbuffer_size.y));
        
        depth = clip_pos0.z * bary.x + clip_pos1.z * bary.y + clip_pos2.z * bary.z;
        uv = v0.texcoord.xy * bary.x + v1.texcoord.xy * bary.y + v2.texcoord.xy * bary.z;
        position = v0.position.xyz * bary.x + v1.position.xyz * bary.y + v2.position.xyz * bary.z;
        position = mul(transform, float4(position, 1.0)).xyz;
        normal = v0.normal.xyz * bary.x + v1.normal.xyz * bary.y + v2.normal.xyz * bary.z;
        normal = v0.normal.xyz * bary.x + v1.normal.xyz * bary.y + v2.normal.xyz * bary.z;
        normal = normalize(mul((float3x3) transform, normal));
        tangent = v0.tangent.xyz * bary.x + v1.tangent.xyz * bary.y + v2.tangent.xyz * bary.z;
        tangent = normalize(mul((float3x3) transform, tangent.xyz));
        dx = v0.texcoord.xy * ddx_barycentrics.x + v1.texcoord.xy * ddx_barycentrics.y + v2.texcoord.xy * ddx_barycentrics.z;
        dy = v0.texcoord.xy * ddy_barycentrics.x + v1.texcoord.xy * ddy_barycentrics.y + v2.texcoord.xy * ddy_barycentrics.z;
        
        if (matID < 1024)
        {
            if (materials[matID].normal_texture < MAX_TEXTURE_ARRAY_SIZE)
            {
                float3 T = tangent;
                float3 B;
                if (!any(T))
                {
                    CreateCoordinateSystem(normal, T, B);
                }
                else
                {
                    float3 B = normalize(cross(normal, T));
                }
        
                float3x3 TBN = float3x3(T, B, normal);
                float3 normalVector = texture_array[NonUniformResourceIndex(materials[matID].normal_texture)].SampleGrad(texture_sampler, uv, dx, dy).rgb;
                normalVector = normalVector * 2.0 - 1.0;
                normalVector = normalize(normalVector);
                normal = normalize(mul(normalVector, TBN));
            }
            
            LoadMaterial();
        }
        
        bitangent = cross(normal, tangent);
        
        if (!front)
        {
            normal *= -1;
            tangent *= -1;
            bitangent *= -1;
        }
    }
};

#endif