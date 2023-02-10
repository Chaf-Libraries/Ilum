#ifndef LIGHT_HLSLI
#define LIGHT_HLSLI

#include "Interaction.hlsli"
#include "Math.hlsli"

#define POINT_LIGHT 0
#define SPOT_LIGHT 1
#define DIRECTIONAL_LIGHT 2
#define RECT_LIGHT 3

#define LIGHT_TYPE_COUNT 4

struct LightInfo
{
    uint point_light_count;
    uint spot_light_count;
    uint directional_light_count;
    uint rect_light_count;
};

struct LightLeSample
{
    float3 L;
    RayDesc ray;
    Interaction isect;
    float pdf_pos;
    float pdf_dir;
    
    void Create(float3 L_, RayDesc ray_, float pdf_pos_, float pdf_dir_)
    {
        L = L_;
        ray = ray_;
        pdf_pos = pdf_pos_;
        pdf_dir = pdf_dir_;
    }

    void Create(float3 L_, RayDesc ray_, Interaction isect_, float pdf_pos_, float pdf_dir_)
    {
        L = L_;
        ray = ray_;
        isect = isect_;
        pdf_pos = pdf_pos_;
        pdf_dir = pdf_dir_;
    }
};

struct LightLiSample
{
    float3 L;
    float3 wi;
    float pdf;
    Interaction isect;
    
    void Create(float3 L_, float3 wi_, float pdf_, Interaction isect_)
    {
        L = L_;
        wi = wi_;
        pdf = pdf_;
        isect = isect_;
    }
};

struct LightSampleContext
{
    float3 p;
    float3 n;
    float3 ns;
};

struct PointLight
{
    float3 color;
    float intensity;
    float3 position;
    
    float filter_scale;
    float light_scale;
    uint filter_sample;
    
    LightLeSample SampleLe(float2 u1, float2 u2, float time)
    {
        RayDesc ray;
        ray.Origin = position;
        ray.Direction = UniformSampleSphere(u1);
        ray.TMin = 0.f;
        ray.TMax = time;
        
        LightLeSample le_sample;
        le_sample.Create(color * intensity, ray, 1, UniformSpherePdf());
        
        return le_sample;
    }
    
    void PDF_Le(RayDesc ray, out float pdf_pos, out float pdf_dir)
    {
        pdf_pos = 0;
        pdf_dir = UniformSpherePdf();
    }

    void PDF_Le(Interaction isect, float3 w, out float pdf_pos, out float pdf_dir)
    {
        
    }

    LightLiSample SampleLi(LightSampleContext ctx, float2 u)
    {
        float3 p = position;
        float3 wi = normalize(p - ctx.p);
        float3 L = color * intensity / DistanceSquared(p, ctx.p);
        
        Interaction isect;
        isect.p = p;
        
        LightLiSample li_sample;
        li_sample.Create(L, wi, 1, isect);
        
        return li_sample;
    }

    float PDF_Li(LightSampleContext ctx, float3 wi)
    {
        return 0.f;
    }
    
    bool IsDelta()
    {
        return true;
    }
};

struct SpotLight
{
    float3 color;
    float intensity;
    float3 position;
    float inner_angle;
    float3 direction;
    float outer_angle;
    float4x4 view_projection;
    
    float filter_scale;
    float light_scale;
    uint filter_sample;
    
    float3 EvalL(float3 p, out float3 wi)
    {
        wi = normalize(position - p);
        float light_angle_scale = 1.0 / max(0.001, cos(inner_angle) - cos(outer_angle));
        float light_angle_offset = -cos(outer_angle) * light_angle_scale;
        float cd = max(dot(-direction, wi), 0.0);
        float angular_attenuation = saturate(cd * light_angle_scale + light_angle_offset);
        return color.rgb * intensity * angular_attenuation * angular_attenuation;
    }
    
    LightLeSample SampleLe(float2 u1, float2 u2, float time)
    {
        // TODO
        
        LightLeSample le_sample;
        
        return le_sample;
    }
    
    void PDF_Le(RayDesc ray, out float pdf_pos, out float pdf_dir)
    {
        // TODO
    }

    void PDF_Le(Interaction isect, float3 w, out float pdf_pos, out float pdf_dir)
    {
        // TODO
    }

    LightLiSample SampleLi(LightSampleContext ctx, float2 u)
    {
        float3 p = position;
        float3 wi;
        float3 L = EvalL(ctx.p, wi) / DistanceSquared(p, ctx.p);
        
        Interaction isect;
        isect.p = p;
        
        LightLiSample li_sample;
        li_sample.Create(L, wi, 1, isect);
        
        return li_sample;
    }

    float PDF_Li(LightSampleContext ctx, float3 wi)
    {
        return 0.f;
    }
    
    bool IsDelta()
    {
        return true;
    }
};

struct DirectionalLight
{
    float3 color;
    float intensity;
    float4 split_depth;
    float4x4 view_projection[4];
    float4 shadow_cam_pos[4];
    float3 direction;
    
    float filter_scale;
    float light_scale;
    uint filter_sample;
    
    LightLiSample SampleLi(LightSampleContext ctx, float2 u)
    {
        float3 wi = normalize(-direction);
        
        Interaction isect;
        isect.p = ctx.p + Infinity * normalize(direction);
        
        float3 L = color.rgb * intensity;
        
        LightLiSample li_sample;
        li_sample.Create(L, wi, 1, isect);
        
        return li_sample;
    }
    
    bool IsDelta()
    {
        return true;
    }
};

struct RectLight
{
    float3 color;
    float intensity;
    float4 corner[4];
    uint texture_id;

    LightLiSample SampleLi(LightSampleContext ctx, float2 u)
    {
        float3 v0 = (corner[1] - corner[0]).xyz;
        float3 v1 = (corner[3] - corner[0]).xyz;
        
        float area = length(cross(v0, v1));
        
        Interaction isect;
        isect.p = corner[0].xyz + v0 * u.x + v1 * u.y;
        
        float3 wi = normalize(isect.p - ctx.p);
        float3 L = color.rgb * intensity / DistanceSquared(isect.p, ctx.p) / area;
        
        LightLiSample li_sample;
        li_sample.Create(L, wi, 1.f / area, isect);
        
        return li_sample;
    }
    
    bool IsDelta()
    {
        return true;
    }
};

#endif