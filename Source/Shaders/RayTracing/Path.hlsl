#include "../RayTracing.hlsli"

struct PathIntegrator
{
    float3 UniformSampleOneLight(inout RayPayload ray_payload, inout Sampler _sampler)
    {
        uint light_count = push_constants.directional_light_count + push_constants.point_light_count + push_constants.spot_light_count + push_constants.area_light_count;
        if (light_count == 0)
        {
            return float3(0.0, 0.0, 0.0);
        }
        
        uint lightNum = (uint) min(light_count - 1, _sampler.Get1D() * (float) light_count);
        float lightPdf = 1.0 / (float) light_count;
        
        Light light;
        light.idx = lightNum;

        float3 wi;
        float light_pdf, scattering_pdf;
        VisibilityTester visibility;
        float3 Li = light.SampleLi(ray_payload.isect, _sampler.Get2D(), wi, light_pdf, visibility);

        float3 Ld = float3(0.0, 0.0, 0.0);
        
        ray_payload.wi = wi;
        if (!IsBlack(Li) && Unoccluded(ray_payload, visibility) && light_pdf != 0.0)
        {
            Ld += ray_payload.f * Li / light_pdf;
        }
        
        // Sample BSDF with multiple importance sampling
        if (!light.IsDelta())
        {
            float3 f = float3(0.0, 0.0, 0.0);
        }
        
        return Ld;
    }
    
    float3 Li(RayDesc ray, inout Sampler _sampler, uint maxDepth)
    {
        float3 radiance = float3(0.0, 0.0, 0.0);
        float3 throughout = float3(1.0, 1.0, 1.0);
                
        for (uint bounce = 0; bounce < maxDepth; bounce++)
        {
            RayPayload ray_payload;
            ray_payload.rnd = _sampler.Get2D();
            ray_payload.isect.ffnormal = float3(0.0, 0.0, 0.0);

            if (!SceneIntersection(ray, ray_payload))
            {
                // Sample environment light
                float pdf = 0;
                radiance += throughout * EnvironmentSampling(ray_payload.isect, _sampler.Get3D(), normalize(ray.Direction), pdf);
                break;
            }
            
            // Sampling next bounce               
            float sample_pdf = ray_payload.pdf;
            float3 sample_f = ray_payload.f;
            float3 sample_wi = ray_payload.wi;
                    
            const float3 n = ray_payload.isect.ffnormal;
            float3 wo = ray_payload.isect.wo;

            radiance += throughout * ray_payload.emission;
            
            // Sampling all lights
            radiance += throughout * UniformSampleOneLight(ray_payload, _sampler);
 
            if (sample_pdf > 0.0 && !IsBlack(sample_f) && abs(dot(sample_wi, ray_payload.isect.ffnormal)) != 0.0)
            {
                throughout *= sample_f * abs(dot(sample_wi, ray_payload.isect.ffnormal)) / sample_pdf;
                ray = SpawnRay(ray_payload.isect, sample_wi);
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