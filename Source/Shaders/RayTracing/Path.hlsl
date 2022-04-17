#include "../RayTracing.hlsli"

struct PathIntegrator
{
    float3 Li(RayDesc ray, inout Sampler _sampler, uint maxDepth)
    {
        uint light_count = push_constants.directional_light_count + push_constants.point_light_count + push_constants.spot_light_count;
        float3 radiance = float3(0.0, 0.0, 0.0);
        float3 throughout = float3(1.0, 1.0, 1.0);
        bool specularBounce = false;
        float etaScale = 1.0;
               
        for (uint bounce = 0; bounce < maxDepth; bounce++)
        {
            Interaction isect;
            isect.normal = float3(0.0, 0.0, 0.0);
            
            bool foundIntersection = SceneIntersection(ray, isect);
           
            if (!foundIntersection)
            {
                float3 w = normalize(ray.Direction);
                radiance += throughout * Skybox.SampleLevel(SkyboxSampler, w, 0.0).rgb;
                break;
            }
            // TODO: Area light sampling

            radiance += throughout * isect.material.emissive;
            
            if (!foundIntersection || bounce >= maxDepth)
            {
                break;
            }
        
            const float3 n = isect.ffnormal;
            float3 wo = isect.wo;
            
            BSDFSampleDesc bsdf;
            bsdf.BxDF_Type = BSDF_ALL;
            bsdf.isect = isect;
            bsdf.woW = wo;
            
            // Sample BSDF to get new path direction
            bsdf.mode = BSDF_Sample;
            bsdf.rnd = _sampler.Get2D();
            CallShader(isect.material.material_type, bsdf);
            
            if (bsdf.bsdf.NumComponents(BSDF_ALL & ~BSDF_SPECULAR) > 0)
            {
                radiance += throughout * UniformSampleOneLight(isect, _sampler, false);
            }

            if (bsdf.pdf > 0.0 && !IsBlack(bsdf.f) && abs(dot(bsdf.wiW, isect.ffnormal)) != 0.0)
            {
                throughout *= bsdf.f * abs(dot(bsdf.wiW, isect.ffnormal)) / bsdf.pdf;
                ray = SpawnRay(isect, bsdf.wiW);
            }
            else
            {
                break;
            }
            
            specularBounce = (bsdf.sampled_type & BSDF_SPECULAR) != 0;
            if ((bsdf.sampled_type & BSDF_SPECULAR) && (bsdf.sampled_type & BSDF_TRANSMISSION))
            {
                float eta = bsdf.eta;
                etaScale *= (dot(bsdf.woW, isect.ffnormal) > 0) ? (eta * eta) : 1.0 / (eta * eta);
            }
            
            float3 rrBeta = throughout * etaScale;
            float max_val = max(rrBeta.x, max(rrBeta.y, rrBeta.z));
            if (max_val < 1.0 && bounce > 3)
            {
                float q = max(0.05, 1.0-max_val);
                if (_sampler.Get1D()<q)
                {
                    break;
                }
                throughout /= 1.0 - q;
            }
        }
        return radiance;
    }
};

[shader("raygeneration")]
void main()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDimensions = DispatchRaysDimensions().xy;
    
    const float2 pixelCenter = float2(launchIndex.xy) + float2(0.5, 0.5);
        
    Sampler sampler_;
    sampler_.Init(launchDimensions, launchIndex, camera.frame_count);
    float2 jitter = (sampler_.Get2D() - float2(0.5, 0.5)) * push_constants.anti_alias;
    float2 screen_coords = (pixelCenter + jitter) / launchDimensions.xy * 2.0 - 1.0;
    
    RayDesc ray = camera.CastRay(screen_coords);
    
    float3 radiance = float3(0.0, 0.0, 0.0);

    PathIntegrator integrator;
    radiance = integrator.Li(ray, sampler_, push_constants.max_bounce);
    
    // Clamp firefly
    float lum = dot(radiance, float3(0.212671, 0.715160, 0.072169));
    if (lum > push_constants.firefly_clamp_threshold)
    {
        radiance *= push_constants.firefly_clamp_threshold / lum;
    }

    int2 launch_id = int2(launchIndex.x, launchDimensions.y - launchIndex.y);
    
    // Temporal Accumulation
    if (camera.frame_count == 0)
    {
        Image[launch_id] = float4(radiance, 1.0);
    }
    else
    {
        float3 prev_color = PrevImage[launch_id].rgb;
        float3 accumulated_color = float3(0.0, 0.0, 0.0);
		
        if ((isnan(prev_color.x) || isnan(prev_color.y) || isnan(prev_color.z)))
        {
            accumulated_color = radiance;
        }
        else
        {
            accumulated_color = lerp(prev_color, radiance, 1.0 / float(camera.frame_count + 1));
        }

        Image[launch_id] = float4(accumulated_color, 1.0);
    }
}