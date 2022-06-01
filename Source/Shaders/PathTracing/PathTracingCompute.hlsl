#include "../ShadingState.hlsli"
#include "../RayTrace.hlsli"
#include "../BxDF.hlsli"

RWTexture2D<float4> shading : register(u0, space0);
cbuffer CameraBuffer : register(b1, space0)
{
    Camera camera;
};
StructuredBuffer<BVHNode> tlas : register(t2, space0);
StructuredBuffer<BVHNode> blas[] : register(t3, space0);

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint3 GroupThreadID : SV_GroupThreadID;
    uint3 GroupID : SV_GroupID;
    uint GroupIndex : SV_GroupIndex;
};

struct Radiance
{
    float3 f_specular;
    float3 f_diffuse;
    float3 f_emissive;
    float3 debug;
};

bool BLASTraversal(RayDesc ray, uint instance_id, out float min_t, out uint primitive_id)
{
    uint node = 0;
    min_t = 1e32;
    float t = 0.0;
    bool find_leaf = false;
    bool terminal = false;

    while (true)
    {
        if (Intersection(blas[instance_id][node].aabb.Transform(instances[instance_id].transform), ray, t))
        {
            if (blas[instance_id][node].left_child == ~0U &&
                blas[instance_id][node].right_child == ~0U)
            {
                // Intersection with triangle
                uint prim_id = blas[instance_id][node].prim_id;
                uint i0 = indices[instances[instance_id].mesh][prim_id * 3 + 0];
                uint i1 = indices[instances[instance_id].mesh][prim_id * 3 + 1];
                uint i2 = indices[instances[instance_id].mesh][prim_id * 3 + 2];
                float3 v0 = mul(instances[instance_id].transform, float4(vertices[instances[instance_id].mesh][i0].position.xyz, 1.0)).xyz;
                float3 v1 = mul(instances[instance_id].transform, float4(vertices[instances[instance_id].mesh][i1].position.xyz, 1.0)).xyz;
                float3 v2 = mul(instances[instance_id].transform, float4(vertices[instances[instance_id].mesh][i2].position.xyz, 1.0)).xyz;
                
                if (Intersection(v0, v1, v2, ray, t))
                {
                    if (min_t > t)
                    {
                        primitive_id = prim_id;
                        min_t = t;
                        find_leaf = true;
                    }
                }
            }
            else
            {
                node = blas[instance_id][node].left_child;
                continue;
            }
        }
        
        while (true)
        {
            uint parent = blas[instance_id][node].parent;
            
            if (parent != ~0U && node == blas[instance_id][parent].left_child)
            {
                if (blas[instance_id][parent].right_child != ~0U)
                {
                    node = blas[instance_id][parent].right_child;
                    break;
                }
            }
            
            node = parent;
            if (node == ~0U)
            {
                terminal = true;
                break;
            }
        }
        
        if (terminal)
        {
            break;
        }
    }
    
    if (!find_leaf)
    {
        return false;
    }
    return true;
}

bool TLASTraversal(RayDesc ray, out uint primtive_id, out uint instance_id)
{
    uint node = 0;
    float min_t = 1e32;
    float t = 0.0;
    bool find_leaf = false;
    bool terminal = false;
    
    while (true)
    {
        if (Intersection(tlas[node].aabb, ray, t))
        {
            if (tlas[node].left_child == ~0U &&
                tlas[node].right_child == ~0U)
            {
                if (min_t > t)
                {
                    float blas_t = 0.0;
                    uint blas_depth = 0;
                    uint prim_id = 0;
                    
                    if (BLASTraversal(ray, tlas[node].prim_id, blas_t, prim_id))
                    {
                        if (min_t > blas_t)
                        {
                            min_t = blas_t;
                            primtive_id = prim_id;
                            instance_id = tlas[node].prim_id;
                            find_leaf = true;
                        }
                    }
                }
            }
            else
            {
                node = tlas[node].left_child;
                continue;
            }
        }
        
        while (true)
        {
            uint parent = tlas[node].parent;
            
            if (parent != ~0U && node == tlas[parent].left_child)
            {
                if (tlas[parent].right_child != ~0U)
                {
                    node = tlas[parent].right_child;
                    break;
                }
            }
            
            node = parent;
            if (node == ~0U)
            {
                terminal = true;
                break;
            }
        }
        
        if (terminal)
        {
            break;
        }
    }
    
    if (!find_leaf)
    {
        return false;
    }
    
    return true;
}

void GetPunctualRadiance(float3 intensity, float3 L, float3 V, ShadingState sstate, inout Radiance radiance)
{
    float alpha_roughness = sstate.mat_info.roughness * sstate.mat_info.roughness;

    float3 H = normalize(L + V);
    
    float NoL = clamp(dot(sstate.normal, L), 0.0, 1.0);
    float NoV = clamp(dot(sstate.normal, V), 0.0, 1.0);
    float NoH = clamp(dot(sstate.normal, H), 0.0, 1.0);
    float LoH = clamp(dot(L, H), 0.0, 1.0);
    float VoH = clamp(dot(V, H), 0.0, 1.0);
    
    if (NoL > 0.0 || NoV > 0.0)
    {
        radiance.f_diffuse += intensity * NoL * Eval_BRDF_Lambertian(sstate.mat_info.F0, sstate.mat_info.F90, sstate.mat_info.c_diff, sstate.mat_info.specular_weight, VoH);
        radiance.f_specular += intensity * NoL * Eval_BRDF_SpecularGGX(sstate.mat_info.F0, sstate.mat_info.F90, alpha_roughness, sstate.mat_info.specular_weight, VoH, NoL, NoV, NoH);
    }
}

[numthreads(32, 32, 1)]
void main(CSParam param)
{
    uint2 extent;
    shading.GetDimensions(extent.x, extent.y);

    if (param.DispatchThreadID.x > extent.x || param.DispatchThreadID.y > extent.y)
    {
        return;
    }
    
    PCGSampler pcg_sampler;
    pcg_sampler.Init(extent, param.DispatchThreadID.xy, camera.frame_count);
          
    float2 jitter = pcg_sampler.Get2D() - 0.5;
    float2 screen_coords = (float2(param.DispatchThreadID.xy) + 0.5 + jitter) / float2(extent);
    screen_coords.y = 1.0 - screen_coords.y;
    screen_coords = screen_coords * 2.0 - 1.0;
    
    RayDesc ray = camera.CastRay(screen_coords);
        
    ShadingState sstate;
    
    uint primtive_id;
    uint instance_id;
    
    float3 radiance = 0;
    float3 throughout = 1.0;
    
    for (uint depth = 0; depth < 3; depth++)
    {
        if (TLASTraversal(ray, primtive_id, instance_id))
        {
            sstate.Load(instance_id, primtive_id, extent, param.DispatchThreadID.xy, camera);
            
            float3 dir = reflect(ray.Direction, sstate.normal);
            ray = SpawnRay(sstate.position, sstate.normal, dir);
            
            radiance += throughout * sstate.mat_info.albedo.rgb;
            throughout *= sstate.mat_info.albedo.rgb;
        }
        else
        {
            break;
        }
    }

    if (camera.frame_count == 0)
    {
        shading[param.DispatchThreadID.xy] = float4(radiance, 1.0);
    }
    else
    {
        float3 prev_color = shading[param.DispatchThreadID.xy].rgb;
        float3 accumulated_color = 0.0;
        if ((isnan(prev_color.x) || isnan(prev_color.y) || isnan(prev_color.z)))
        {
            accumulated_color = radiance;
        }
        else
        {
            accumulated_color = lerp(prev_color, radiance, 1.0 / float(camera.frame_count + 1));
        }
        shading[param.DispatchThreadID.xy] = float4(accumulated_color, 1.0);
    }

}