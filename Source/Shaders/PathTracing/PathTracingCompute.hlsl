#include "../ShadingState.hlsli"
#include "../RayTrace.hlsli"
#include "../BxDF.hlsli"
#include "../Tonemapper.hlsli"
#include "../Lights.hlsli"

RWTexture2D<float4> shading : register(u0, space0);
cbuffer CameraBuffer : register(b1, space0)
{
    Camera camera;
};
StructuredBuffer<BVHNode> tlas : register(t2, space0);
StructuredBuffer<BVHNode> blas[] : register(t3, space0);

[[vk::push_constant]]
struct
{
    uint directional_light_count;
    uint spot_light_count;
    uint point_light_count;
    uint area_light_count;
} push_constants;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint3 GroupThreadID : SV_GroupThreadID;
    uint3 GroupID : SV_GroupID;
    uint GroupIndex : SV_GroupIndex;
};

bool BLASTraversal(RayDesc ray, uint instance_id, out float min_t, out uint primitive_id, out float3 bary)
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
                
                float3 barycentric;
                if (Intersection(v0, v1, v2, ray, t, barycentric))
                {
                    if (min_t > t)
                    {
                        primitive_id = prim_id;
                        min_t = t;
                        bary = barycentric;
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

bool TLASTraversal(RayDesc ray, out uint primtive_id, out uint instance_id, out float3 bary)
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
                    float3 barycentric = 0;
                    
                    if (BLASTraversal(ray, tlas[node].prim_id, blas_t, prim_id, barycentric))
                    {
                        if (min_t > blas_t)
                        {
                            min_t = blas_t;
                            primtive_id = prim_id;
                            instance_id = tlas[node].prim_id;
                            bary = barycentric;
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

bool TraversalShadowRayBLAS(RayDesc ray, uint instance_id)
{
    uint node = 0;
    float t = 0.0;

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
                
                float3 bary;
                if (Intersection(v0, v1, v2, ray, t, bary))
                {
                    if (t < ray.TMax)
                    {
                        return true;
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
                return false;
            }
        }
    }
}

bool TraversalShadowRayTLAS(RayDesc ray)
{
    uint node = 0;
    float t = 0.0;
    
    while (true)
    {
        if (Intersection(tlas[node].aabb, ray, t))
        {
            if (tlas[node].left_child == ~0U &&
                tlas[node].right_child == ~0U)
            {
                if (t < ray.TMax && TraversalShadowRayBLAS(ray, tlas[node].prim_id))
                {
                    return true;
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
                return false;
            }
        }
    }
}

struct Light
{
    uint light_num;
    VisibilityTester vis;
    float3 L;
    float pdf;
    
    float3 Eval(float3 shading_point, out float3 L)
    {
        if (light_num < push_constants.directional_light_count)
        {
            DirectionalLight light = directional_lights[light_num];
            return Eval_Light(light, shading_point, L);
        }
        
        if (light_num < push_constants.directional_light_count + push_constants.point_light_count)
        {
            PointLight light = point_lights[light_num - push_constants.directional_light_count];
            return Eval_Light(light, shading_point, L);
        }
        
        if (light_num < push_constants.directional_light_count + push_constants.point_light_count + push_constants.spot_light_count)
        {
            SpotLight light = spot_lights[light_num - push_constants.directional_light_count - push_constants.point_light_count];
            return Eval_Light(light, shading_point, L);
        }
        
        return 0.0;
    }
    
    float3 Sample(float3 shading_point, float2 pcg)
    {
        if (light_num < push_constants.directional_light_count)
        {
            DirectionalLight light = directional_lights[light_num];
            return Sample_Light(light, shading_point, pcg, L, pdf, vis);
        }
        
        if (light_num < push_constants.directional_light_count + push_constants.point_light_count)
        {
            PointLight light = point_lights[light_num - push_constants.directional_light_count];
            return Sample_Light(light, shading_point, pcg, L, pdf, vis);
        }
        
        if (light_num < push_constants.directional_light_count + push_constants.point_light_count + push_constants.spot_light_count)
        {
            SpotLight light = spot_lights[light_num - push_constants.directional_light_count - push_constants.point_light_count];
            return Sample_Light(light, shading_point, pcg, L, pdf, vis);
        }
        
        return 0.0;
    }
    
    bool IsOccluded(float3 normal)
    {
        RayDesc ray;
        ray.Origin = OffsetRay(vis.from, dot(vis.direction, normal) > 0 ? normal : -normal);
        ray.Direction = normalize(vis.direction);
        ray.TMin = 0.0;
        ray.TMax = vis.distance;
        
        return TraversalShadowRayTLAS(ray);
    }
};

float3 Eval_BRDF(float3 L, float3 V, ShadingState sstate)
{
    float alpha_roughness = sstate.mat_info.roughness * sstate.mat_info.roughness;

    float3 H = normalize(L + V);
    
    float NoL = clamp(dot(sstate.normal, L), 0.0, 1.0);
    float NoV = clamp(dot(sstate.normal, V), 0.0, 1.0);
    float NoH = clamp(dot(sstate.normal, H), 0.0, 1.0);
    float LoH = clamp(dot(L, H), 0.0, 1.0);
    float VoH = clamp(dot(V, H), 0.0, 1.0);
    
    float3 diffuse = 0.0;
    float3 specular = 0.0;
    
    if (NoL > 0.0 || NoV > 0.0)
    {
        diffuse = NoL * Eval_BRDF_Lambertian(sstate.mat_info.F0, sstate.mat_info.F90, sstate.mat_info.c_diff, sstate.mat_info.specular_weight, VoH);
        specular = NoL * Eval_BRDF_SpecularGGX(sstate.mat_info.F0, sstate.mat_info.F90, alpha_roughness, sstate.mat_info.specular_weight, VoH, NoL, NoV, NoH);
    }
    
    return sstate.mat_info.emissive + diffuse + specular;
}

float BRDF_Pdf(ShadingState sstate, float NoH, float3 L, inout PCGSampler pcg_sampler)
{
    float diffuse_ratio = 0.5 * (1.0 - sstate.mat_info.metallic);
    if (pcg_sampler.Get1D() < diffuse_ratio)
    {
        return dot(sstate.normal, L) * InvPI * diffuse_ratio;
    }
    else
    {
        float alpha = sstate.mat_info.roughness * sstate.mat_info.roughness;
        return D_GGX(NoH, alpha) / 4.0 * (1 - diffuse_ratio);
    }
}


float3 SampleBRDF(float3 V, ShadingState sstate, inout PCGSampler pcg_sampler, out float3 L, out float pdf)
{
    float diffuse_ratio = 0.5 * (1.0 - sstate.mat_info.metallic);
    diffuse_ratio = 1.0;
    //if (pcg_sampler.Get1D() < diffuse_ratio)
    {
        // Diffuse term
        L = SampleCosineHemisphere(pcg_sampler.Get2D());
        L = sstate.tangent * L.x + sstate.bitangent * L.y + sstate.normal * L.z;
        
        float3 H = normalize(L + V);
        float VoH = clamp(dot(V, H), 0.0, 1.0);
        
        if (dot(sstate.normal, L) < 0.0)
        {
            return 0.0;
        }
        
        pdf = dot(sstate.normal, L) * InvPI * diffuse_ratio;
        return Eval_BRDF_Lambertian(sstate.mat_info.F0, sstate.mat_info.F90, sstate.mat_info.c_diff, sstate.mat_info.specular_weight, VoH);
    }
    //else
    //{
    //    // Specular term
    //    float alpha = sstate.mat_info.roughness * sstate.mat_info.roughness;
    //    float2 xi = pcg_sampler.Get2D();
    //    float cos_theta = saturate(sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y)));
    //    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    //    float phi = 2.0 * PI * xi.x;
    //    float3 H = normalize(float3(
    //        sin_theta * cos(phi),
    //        sin_theta * sin(phi),
    //        cos_theta));
    //    H = sstate.tangent * H.x + sstate.bitangent * H.y + sstate.normal * H.z;
    //    
    //    float VoH = clamp(dot(V, H), 0.0, 1.0);
    //    float NoL = clamp(dot(sstate.normal, L), 0.0, 1.0);
    //    float NoV = clamp(dot(sstate.normal, V), 0.0, 1.0);
    //    float NoH = clamp(dot(sstate.normal, H), 0.0, 1.0);
    //    
    //    L = normalize(reflect(-V, H));
    //    
    //    if (dot(sstate.normal, L) < 0.0)
    //    {
    //        return 0.0;
    //    }
    //    
    //    pdf = D_GGX(cos_theta, alpha) / 4.0 * (1 - diffuse_ratio);
    //    return Eval_BRDF_SpecularGGX(sstate.mat_info.F0, sstate.mat_info.F90, alpha, sstate.mat_info.specular_weight, VoH, NoL, NoV, NoH);
    //}
}

float3 DirectLight(ShadingState sstate, inout PCGSampler pcg_sampler)
{
    uint light_count = push_constants.directional_light_count + push_constants.point_light_count + push_constants.spot_light_count + push_constants.area_light_count;
    if (light_count > 0)
    {
        Light light;

        light.light_num = (uint) min(light_count - 1, pcg_sampler.Get1D() * (float) light_count);
        float light_pdf = 1.0 / (float) light_count;
        
        float3 Li = light.Sample(sstate.position, pcg_sampler.Get2D());
        float3 Ld = 0;
        
        if (!light.IsOccluded(sstate.normal) && light.pdf != 0.0)
        {
            float3 V = normalize(camera.position - sstate.position);
            float3 f = Eval_BRDF(light.L, V, sstate) * abs(dot(light.L, sstate.normal));
            return f * Li / light.pdf / light_pdf;
        }
    }
        
    return 0.0;
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
    
    for (uint bounce = 0; bounce < 3; bounce++)
    {
        float3 bary;
        if (TLASTraversal(ray, primtive_id, instance_id, bary))
        {
            sstate.Load(instance_id, primtive_id, bary, camera);
            sstate.normal = dot(sstate.normal, ray.Direction) <= 0 ? sstate.normal : -sstate.normal;
            CreateCoordinateSystem(sstate.normal, sstate.tangent, sstate.bitangent);
            
            float3 V = normalize(camera.position - sstate.position);
            float3 L = 0;
            float pdf = 0;
            
            radiance += throughout * DirectLight(sstate, pcg_sampler);
            float3 f = SampleBRDF(V, sstate, pcg_sampler, L, pdf);
            
            if (pdf > 0.0 && dot(L, sstate.normal) != 0.0 && any(f) != 0.0)
            {
                throughout *= f * abs(dot(L, sstate.normal)) / pdf;
                ray = SpawnRay(sstate.position, sstate.normal, normalize(L));
            }
            else
            {
                break;
            }
        }
        else
        {
            break;
        }
        
        // Russian roulette
        float3 rr_beta = throughout;
        float max_cmpt = max(rr_beta.x, max(rr_beta.y, rr_beta.z));
        if (max_cmpt < 1.0 && bounce > 3)
        {
            float q = max(0.05, 1.0 - max_cmpt);
            if (pcg_sampler.Get1D() < q)
            {
                break;
            }
            throughout /= 1.0 - q;
        }
    }
    
    // Clamp firefly
    float lum = Luminance(radiance);
    if (lum > 4.0)
    {
        radiance *= 4.0 / lum;
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