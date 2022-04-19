#include "../RayTracing.hlsli"

#ifndef RUNTIME
#define RAYGEN
#define CLOSEST_HIT
#define MISS
#endif

#ifdef RAYGEN
[shader("raygeneration")]
void RayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDimensions = DispatchRaysDimensions().xy;
    
    const float2 pixelCenter = float2(launchIndex.xy) + float2(0.5, 0.5);
    
    RayPayload ray_payload;
    ray_payload._sampler.Init(launchDimensions, launchIndex, camera.frame_count);

    float2 jitter = (ray_payload._sampler.Get2D() - float2(0.5, 0.5)) * push_constants.anti_alias;
    float2 screen_coords = (pixelCenter + jitter) / launchDimensions.xy * 2.0 - 1.0;
    
    RayDesc ray = camera.CastRay(screen_coords);

    ray_payload.radiance = float3(0.0, 0.0, 0.0);
    ray_payload.throughout = float3(1.0, 1.0, 1.0);
    ray_payload.eta = 1.0;
    ray_payload.terminate = false;
                
    for (uint bounce = 0; bounce < push_constants.max_bounce && !ray_payload.terminate; bounce++)
    {
        if (!SceneIntersection(ray, ray_payload))
        {
            // Sample environment light
            float pdf = 0;
            ray_payload.radiance += ray_payload.throughout * EnvironmentSampling(ray_payload, normalize(ray.Direction), pdf);
            break;
        }
           
        ray = SpawnRay(ray_payload.isect, ray_payload.wi);
        
        // Russian roulette
        float3 rrBeta = ray_payload.throughout * ray_payload.eta;
        float max_cmpt = max(rrBeta.x, max(rrBeta.y, rrBeta.z));
        if(max_cmpt<1.0&&bounce>3)
        {
            float q = max(0.05, 1.0 - max_cmpt);
            if (ray_payload._sampler.Get1D()<q)
            {
                break;
            }
            ray_payload.throughout /= 1.0 - q;
        }
    }
    
    // Clamp firefly
    float lum = dot(ray_payload.radiance, float3(0.212671, 0.715160, 0.072169));
    if (lum > push_constants.firefly_clamp_threshold)
    {
        ray_payload.radiance *= push_constants.firefly_clamp_threshold / lum;
    }
    
    int2 launch_id = int2(launchIndex.x, launchDimensions.y - launchIndex.y);
    
    // Temporal Accumulation
    if (camera.frame_count == 0)
    {
        Image[launch_id] = float4(ray_payload.radiance, 1.0);
    }
    else
    {
        float3 prev_color = PrevImage[launch_id].rgb;
        float3 accumulated_color = float3(0.0, 0.0, 0.0);
		
        if ((isnan(prev_color.x) || isnan(prev_color.y) || isnan(prev_color.z)))
        {
            accumulated_color = ray_payload.radiance;
        }
        else
        {
            accumulated_color = lerp(prev_color, ray_payload.radiance, 1.0 / float(camera.frame_count + 1));
        }

        Image[launch_id] = float4(accumulated_color, 1.0);
    }
}
#endif

#ifdef CLOSEST_HIT
[shader("closesthit")]
void Closesthit(inout RayPayload ray_payload : SV_RayPayload, BuiltInTriangleIntersectionAttributes Attributes)
{
    ray_payload.ishit = true;
        
    RayDesc ray;
    ray.Direction = WorldRayDirection();
    ray.Origin = WorldRayOrigin();
    
    GetInteraction(ray_payload, ray, Attributes);

    Material material;
    GetMaterial(ray_payload.isect, material, ray, InstanceIndex());
    
    // Setup BSDF
    BSDFs bsdf;
#ifdef USE_Matte
    bsdf = CreateMatteMaterial(material);
#endif
#ifdef USE_Metal
    bsdf = CreateMetalMaterial(material);
#endif
#ifdef USE_Plastic
    bsdf = CreatePlasticMaterial(material);
#endif
#ifdef USE_Glass
    bsdf = CreateGlassMaterial(material);
#endif
#ifdef USE_Mirror
    bsdf = CreateMirrorMaterial(material);
#endif
#ifdef USE_Substrate
    bsdf = CreateSubstrateMaterial(material);
#endif
#ifdef USE_Disney
    bsdf = CreateDisneyMaterial(material);
#endif
    bsdf.isect = ray_payload.isect;
    
    // Sample emission
    ray_payload.radiance += material.emissive;
    
    // Uniform sample one light
    uint light_count = push_constants.directional_light_count + push_constants.point_light_count + push_constants.spot_light_count + push_constants.area_light_count;
    if (light_count > 0)
    {
        uint lightNum = (uint) min(light_count - 1, ray_payload._sampler.Get1D() * (float) light_count);
        float lightPdf = 1.0 / (float) light_count;
        
        Light light;
        light.idx = lightNum;

        // Sample light source with multiple importance sampling
        float3 wi;
        float light_pdf, scattering_pdf;
        VisibilityTester visibility;
        float3 Li = light.SampleLi(ray_payload.isect, ray_payload._sampler.Get2D(), wi, light_pdf, visibility);

        float3 Ld = float3(0.0, 0.0, 0.0);
        
        if (!IsBlack(Li) && Unoccluded(ray_payload, visibility) && light_pdf != 0.0)
        {
            // Evaluate BSDF
            float3 f = bsdf.f(ray_payload.isect.wo, wi, BSDF_ALL) * abs(dot(wi, ray_payload.isect.ffnormal));
            scattering_pdf = bsdf.Pdf(ray_payload.isect.wo, wi, BSDF_ALL);
            // Add light's contribution to reflected radiance
            if (light.IsDelta())
            {
                Ld += f * Li / light_pdf;
            }
            else
            {
                // Light MIS
                float weight = PowerHeuristic(1, light_pdf, 1, scattering_pdf);
                Ld += f * Li * weight / light_pdf;
            }
        }

        // Sample BSDF with multiple importance sampling
        if (!light.IsDelta())
        {
            uint sample_type;
            float3 f = bsdf.Samplef(ray_payload.isect.wo, ray_payload._sampler.Get2D(), wi, scattering_pdf, BSDF_ALL, sample_type);

            if(!IsBlack(f)&&scattering_pdf>0.0)
            {
                float weight = 1.0;
                RayDesc r = SpawnRay(ray_payload.isect, wi);
                float3 p;
                if (light.IsIntersect(r, p))
                {
                    visibility.from = ray_payload.isect;
                    visibility.dir = wi;
                    visibility.dist = length(p - ray_payload.isect.position);
                    if (Unoccluded(ray_payload, visibility))
                    {
                        float pdf_li = dot(ray_payload.isect.position - p, ray_payload.isect.position - p) / abs(dot(ray_payload.isect.ffnormal, wi)) * light.Area();
                        if (pdf_li != 0.0)
                        {
                            weight = PowerHeuristic(1, scattering_pdf, 1, light_pdf);
                            Ld += light.Le() * f * weight / scattering_pdf;
                        }
                    }
                }
            }
            
        }
        
        // Add direct light
        ray_payload.radiance += Ld / lightPdf * ray_payload.throughout;
    }
        
    // Sample BSDF to get new path direction
    float3 wi;
    float pdf;
    uint sample_type;
    float3 f = bsdf.Samplef(ray_payload.isect.wo, ray_payload._sampler.Get2D(), wi, pdf, BSDF_ALL, sample_type);
    if (IsBlack(f) || pdf == 0.0)
    {
        ray_payload.terminate = true;
        return;
    }
    
    if ((sample_type & BSDF_SPECULAR) && (sample_type & BSDF_TRANSMISSION))
    {
        float eta = bsdf.eta;
        ray_payload.eta *= (dot(ray_payload.isect.wo, ray_payload.isect.ffnormal) > 0) ? (eta * eta) : 1 / (eta * eta);
    }
    
    if (pdf > 0.0 && !IsBlack(f) && abs(dot(wi, ray_payload.isect.ffnormal)) != 0.0)
    {
        ray_payload.throughout *= f * abs(dot(wi, ray_payload.isect.ffnormal)) / pdf;
        ray_payload.wi = wi;
    }
    else
    {
        ray_payload.terminate = true;
        return;
    }
}
#endif

#ifdef MISS
[shader("miss")]
void Miss(inout RayPayload ray_payload : SV_RayPayload)
{
    ray_payload.visibility = true;
}
#endif