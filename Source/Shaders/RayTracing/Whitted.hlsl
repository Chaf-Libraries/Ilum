#include "../RayTracing.hlsli"

struct WhittedIntegrator
{
    float3 Li(RayDesc ray, Sampler _sampler, uint maxDepth)
    {
        float3 radiance = float3(0.0, 0.0, 0.0);
        float3 throughout = float3(1.0, 1.0, 1.0);
        float3 absorption = float3(0.0, 0.0, 0.0);
        
        Interaction isect;

        for (uint bounce = 0; bounce < maxDepth; bounce++)
        {
            // Hit miss
            if (!SceneIntersection(ray, isect))
            {
                // Sample environment light
                float3 w = normalize(ray.Direction);
                radiance += throughout * Skybox.SampleLevel(SkyboxSampler, w, 0.0).rgb;
                break;
            }
        
            const float3 n = isect.ffnormal;
            float3 wo = isect.wo;
                    
            // TODO: Sampling emissive
            // TODO: Handle area light
            
            // Sampling Lights
            // Sampling Point Light
            for (uint i = 0; i < push_constants.point_light_count; i++)
            {
                float3 wi;
                float pdf;
                PointLight light = point_lights[i];
                float3 Li = light.SampleLi(isect, _sampler.Get2D(), wi, pdf);
                if (IsBlack(Li) || pdf == 0)
                {
                    continue;
                }
                
                float3 dir = normalize(light.position - isect.position);
                float dist = length(light.position - isect.position);
                
                if (!IsBlack(Li) && VisibilityTest(isect, dir, dist))
                {
                    //wi = isect.WorldToLocalDir(wi);
                    BSDF bsdf;
                    bsdf.Init(isect);
                    float3 f = bsdf.f(wo, wi);
                    //wi = isect.LocalToWorldDir(wi);
                    radiance += throughout * f * Li * abs(dot(wi, isect.ffnormal)) / pdf;
                }
            }
            
            // Sampling Directional Light
            for (i = 0; i < push_constants.directional_light_count; i++)
            {
                float3 wi;
                float pdf;
                DirectionalLight light = directional_lights[i];
                float3 Li = light.SampleLi(isect, _sampler.Get2D(), wi, pdf);
                if (IsBlack(Li) || pdf == 0)
                {
                    continue;
                }
                
                float3 dir = -light.direction;
                float dist = Infinity;
                
                if (!IsBlack(Li) && VisibilityTest(isect, dir, dist))
                {
                    wi = isect.WorldToLocalDir(wi);
                    BSDF bsdf;
                    bsdf.Init(isect);
                    float3 f = bsdf.f(wo, wi);
                    wi = isect.LocalToWorldDir(wi);
                    radiance += throughout * f * Li * abs(dot(wi, isect.ffnormal)) / pdf;
                }
            }
            
            // Sampling Point Light
            for (i = 0; i < push_constants.spot_light_count; i++)
            {
                float3 wi;
                float pdf;
                SpotLight light = spot_lights[i];
                float3 Li = light.SampleLi(isect, _sampler.Get2D(), wi, pdf);
                if (IsBlack(Li) || pdf == 0)
                {
                    continue;
                }
                
                float3 dir = normalize(light.position - isect.position);
                float dist = length(light.position - isect.position);
                
                if (!IsBlack(Li) && VisibilityTest(isect, dir, dist))
                {
                    wi = isect.WorldToLocalDir(wi);
                    BSDF bsdf;
                    bsdf.Init(isect);
                    float3 f = bsdf.f(wo, wi);
                    wi = isect.LocalToWorldDir(wi);
                    radiance += throughout * f * Li * abs(dot(wi, isect.ffnormal)) / pdf;
                }
            }
            
            // Sampling next bounce
            float3 wi;
            float pdf;
		    
            BSDF bsdf;
            bsdf.Init(isect);
            float3 f = bsdf.Samplef(wo, _sampler, wi, pdf);
            //wi = isect.LocalToWorld(wi);
            
            //radiance = f;

            if (pdf > 0.0 && !IsBlack(f) && abs(dot(wi, isect.ffnormal)) != 0.0)
            {
                throughout *= f * abs(dot(wi, isect.ffnormal)) / pdf;
                ray.Direction = wi;
                ray.Origin = OffsetRay(isect.position, dot(wi, isect.ffnormal) > 0.0 ? isect.ffnormal : -isect.ffnormal);
            }
            else
            {
			    break;
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

    WhittedIntegrator integrator;
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
            accumulated_color = lerp(prev_color, radiance, 1.0 / float(camera.frame_count));
        }

        Image[launch_id] = float4(accumulated_color, 1.0);
    }
}