#ifndef __SHADINGSTATE_HLSL__
#define __SHADINGSTATE_HLSL__

#include "Common.hlsli"
#include "Math.hlsli"

// Ray = O+td
struct RayDifferential
{
    float3 dOdx;
    float3 dOdy;
    float3 dDdx;
    float3 dDdy;
};

void ComputeRayDirectionDifferentials(float3 ray_dir, float3 right, float3 up, float2 viewport_dims, out float3 dDdx, out float3 dDdy)
{
    float dd = dot(ray_dir, ray_dir);
    float divd = 2.0 / (dd * sqrt(dd));
    float dr = dot(ray_dir, right);
    float du = dot(ray_dir, up);
    dDdx = ((dd * right) - (dr * ray_dir)) * divd / viewport_dims.x;
    dDdy = -((dd * up) - (du * ray_dir)) * divd / viewport_dims.y;
}

void PropagateRayDifferential(float3 direction, float t, float3 normal, inout RayDifferential rd)
{
    float3 dodx = rd.dOdx + t * rd.dDdx;
    float3 dody = rd.dOdy + t * rd.dDdy;
    float rcpDN = 1.0 / dot(direction, normal);
    float dtdx = -dot(dodx, normal) * rcpDN;
    float dtdy = -dot(dody, normal) * rcpDN;
    dodx += direction * dtdx;
    dody += direction * dtdy;
    rd.dOdx = dodx;
    rd.dOdy = dody;
}

void PrepVerticesForRayDifferentials(Vertex vertices[3], float4x4 transform, out float3 edge01, out float3 edge02, out float3 face_normal)
{
    float3 v0 = mul(vertices[0].position.xyz, (float3x3) transform);
    float3 v1 = mul(vertices[1].position.xyz, (float3x3) transform);
    float3 v2 = mul(vertices[2].position.xyz, (float3x3) transform);
    
    edge01 = v1 - v0;
    edge02 = v2 - v0;
    face_normal = cross(edge01, edge02);
}

void InterpolateTexCoordDifferentials(float2 dBarydx, float2 dBarydy, Vertex vertices[3], out float2 dx, out float2 dy)
{
    float2 delta1 = vertices[1].texcoord.xy - vertices[0].texcoord.xy;
    float2 delta2 = vertices[2].texcoord.xy - vertices[0].texcoord.xy;
    dx = dBarydx.x * delta1 + dBarydx.y * delta2;
    dy = dBarydy.x * delta1 + dBarydy.y * delta2;
}

void ComputeBarycentricDifferentials(RayDifferential rd, float3 ray_dir, float3 edge01, float3 edge02, float3 face_normal, out float2 dBarydx, out float2 dBarydy)
{
    float3 Nu = cross(edge01, face_normal);
    float3 Nv = cross(edge02, face_normal);
    
    float3 Lu = Nu / dot(Nu, edge01);
    float3 Lv = Nv / dot(Nv, edge02);

    dBarydx.x = dot(Lu, rd.dOdx);
    dBarydx.y = dot(Lv, rd.dOdx);
    dBarydy.x = dot(Lu, rd.dOdy);
    dBarydy.y = dot(Lv, rd.dOdy);
}

void ComputeUVDifferentials(Vertex vertices[3], float4x4 transform, float3 ray_dir, float hit_t, Camera camera, float2 viewport_dims, out float2 dUVdx, out float2 dUVdy)
{
    // Initialize
    RayDifferential rd = (RayDifferential) 0;
    
    // Get ray direction differentials
    ComputeRayDirectionDifferentials(ray_dir, camera.right.xyz, camera.up.xyz, viewport_dims, rd.dDdx, rd.dDdy);
    
    // Get triangle edges and face normal
    float3 edge01, edge02, face_normal;
    PrepVerticesForRayDifferentials(vertices, transform, edge01, edge02, face_normal);
    
    // Propagate the ray differential to the current hit point
    PropagateRayDifferential(ray_dir, hit_t, face_normal, rd);
    
    float2 dBarydx, dBarydy;
    ComputeBarycentricDifferentials(rd, ray_dir, edge01, edge02, face_normal, dBarydx, dBarydy);

    InterpolateTexCoordDifferentials(dBarydx, dBarydy, vertices, dUVdx, dUVdy);
}

static const uint MetalRoughnessWorkflow = 0;
static const uint SpecularGlossinessWorkflow = 1;

#define MAX_TEXTURE_ARRAY_SIZE 1024

struct MaterialInfo
{
    float4 albedo;
    float roughness;
    float metallic;
    
    
    float ior;
    
    float3 f0; // full reflectance color
    float3 f90; // reflectance color at grazing angle
    float3 c_diff;
};

struct ShadingState
{
    RayDesc ray;
    RayDifferential ray_differential;
    
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
            mat_info.albedo = materials[matID].pbr_diffuse_factor;
            if (materials[matID].pbr_diffuse_texture < MAX_TEXTURE_ARRAY_SIZE)
            {
                mat_info.albedo *= texture_array[materials[matID].pbr_diffuse_texture].SampleGrad(texture_sampler, uv, dx, dy);
            }
            if (materials[matID].pbr_specular_glossiness_texture < MAX_TEXTURE_ARRAY_SIZE)
            {
                float4 sg_sample = texture_array[materials[matID].pbr_specular_glossiness_texture].SampleGrad(texture_sampler, uv, dx, dy);
                mat_info.roughness *= sg_sample.a; // glossiness to roughness
                mat_info.f0 *= sg_sample.rgb; // specular
            }
            mat_info.roughness = 1.0 - mat_info.roughness;
            mat_info.c_diff = mat_info.albedo.rgb * (1.0 - max(max(mat_info.f0.r, mat_info.f0.g), mat_info.f0.b));
        }
        
        if (materials[matID].type == MetalRoughnessWorkflow)
        {
            mat_info.metallic = materials[matID].pbr_metallic_factor;
            mat_info.roughness = materials[matID].pbr_glossiness_factor;
            mat_info.albedo = materials[matID].pbr_base_color_factor;
            if (materials[matID].pbr_base_color_texture < MAX_TEXTURE_ARRAY_SIZE)
            {
                mat_info.albedo *= texture_array[materials[matID].pbr_base_color_texture].SampleGrad(texture_sampler, uv, dx, dy);
            }
            if (materials[matID].pbr_metallic_roughness_texture < MAX_TEXTURE_ARRAY_SIZE)
            {
                float4 mr_sample = texture_array[materials[matID].pbr_specular_glossiness_texture].SampleGrad(texture_sampler, uv, dx, dy);
                mat_info.roughness *= mr_sample.g;
                mat_info.metallic *= mr_sample.b;
            }
            mat_info.c_diff = lerp(mat_info.albedo.rgb, float3(0.0, 0.0, 0.0), mat_info.metallic);
            mat_info.f0 = lerp(mat_info.f0, mat_info.albedo.rgb, mat_info.metallic);
        }
    }
    
    void LoadVisibilityBuffer(Texture2D<uint> vbuffer, uint2 pixel_coord, Camera cam)
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
        
        float4 clip_pos0 = mul(cam.view_projection, mul(transform, float4(v0.position.xyz, 1.0)));
        float4 clip_pos1 = mul(cam.view_projection, mul(transform, float4(v1.position.xyz, 1.0)));
        float4 clip_pos2 = mul(cam.view_projection, mul(transform, float4(v2.position.xyz, 1.0)));
        
        float2 p0 = clip_pos1.xy - clip_pos0.xy;
        float2 p1 = clip_pos0.xy - clip_pos2.xy;
        
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
        
        depth = clip_pos0.z * bary.x + clip_pos1.z * bary.y + clip_pos2.z * bary.z;
        uv = v0.texcoord.xy * bary.x + v1.texcoord.xy * bary.y + v2.texcoord.xy * bary.z;
        position = v0.position.xyz * bary.x + v1.position.xyz * bary.y + v2.position.xyz * bary.z;
        position = mul(transform, float4(position, 1.0)).xyz;
        normal = v0.normal.xyz * bary.x + v1.normal.xyz * bary.y + v2.normal.xyz * bary.z;
        normal = v0.normal.xyz * bary.x + v1.normal.xyz * bary.y + v2.normal.xyz * bary.z;
        normal = normalize(mul((float3x3) transform, normal));
        tangent = v0.tangent.xyz * bary.x + v1.tangent.xyz * bary.y + v2.tangent.xyz * bary.z;
        tangent = normalize(mul((float3x3) transform, tangent.xyz));
        
        float3 ray_dir = normalize(position - cam.position);
        float hit_t = length(position - cam.position);
        Vertex vertex[3] = { v0, v1, v2 };
        
        ComputeUVDifferentials(vertex, transform, ray_dir, hit_t, cam, float2(vbuffer_size), dx, dy);
        
        bool double_sided = false;
        
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
            double_sided = materials[matID].double_sided == 1;
        }
        
        bitangent = cross(normal, tangent);
        
        if (double_sided && dot(cam.position - position, normal) < 0.0)
        {
            normal *= -1;
            tangent *= -1;
            bitangent *= -1;
        }
    }
};

#endif