#ifndef LIGHT_HLSLI
#define LIGHT_HLSLI

#include "Interaction.hlsli"
#include "Math.hlsli"

#define POINT_LIGHT 0
#define SPOT_LIGHT 1
#define DIRECTIONAL_LIGHT 2
#define RECT_LIGHT 3

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
    float radius;
    float range;
    
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

    float3 Li(float3 p, out float3 wi)
    {
        wi = normalize(position - p);
        float d = length(position - p);
        float attenuation = max(min(1.0 - pow(d / range, 4.0), 1.0), 0.0) / (d * d);
        return color.rgb * intensity * attenuation;
    }
    
    bool IsDelta()
    {
        return true;
    }
    
    float3 SampleLi(SurfaceInteraction interaction, float2 u, out float3 wi, out float pdf, out VisibilityTester visibility)
    {
        wi = normalize(position - interaction.isect.p);
        pdf = 1.0;

        float d = length(position - interaction.isect.p);
        float attenuation = max(min(1.0 - pow(d / range, 4.0), 1.0), 0.0) / (d * d);
        
        visibility.from = interaction;
        visibility.dir = normalize(position - interaction.isect.p);
        visibility.dist = length(position - interaction.isect.p);
        
        return color.rgb * intensity * attenuation;
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

    float3 Li(float3 p, out float3 wi)
    {
        wi = normalize(position - p);
        float light_angle_scale = 1.0 / max(0.001, cos(inner_angle) - cos(outer_angle));
        float light_angle_offset = -cos(outer_angle) * light_angle_scale;
        float cd = max(dot(direction, wi), 0.0);
        float angular_attenuation = saturate(cd * light_angle_scale + light_angle_offset);
        return color.rgb * intensity * angular_attenuation * angular_attenuation;
    }
    
    bool IsDelta()
    {
        return true;
    }
    
    float3 SampleLi(SurfaceInteraction interaction, float2 u, out float3 wi, out float pdf, out VisibilityTester visibility)
    {
        wi = normalize(position - interaction.isect.p);
        pdf = 1.0;

        float light_angle_scale = 1.0 / max(0.001, cos(inner_angle) - cos(outer_angle));
        float light_angle_offset = -cos(outer_angle) * light_angle_scale;
        float cd = max(dot(direction, wi), 0.0);
        float angular_attenuation = saturate(cd * light_angle_scale + light_angle_offset);

        visibility.from = interaction;
        visibility.dir = normalize(position - interaction.isect.p);
        visibility.dist = length(position - interaction.isect.p);
        
        return color.rgb * intensity * angular_attenuation * angular_attenuation;
    }
};

struct DirectionalLight
{
    float3 color;
    float intensity;
    float3 direction;

    float3 Li(float3 p, out float3 wi)
    {
        wi = normalize(-direction);
        return color.rgb * intensity;
    }
    
    bool IsDelta()
    {
        return true;
    }
    
    float3 SampleLi(SurfaceInteraction interaction, float2 u, out float3 wi, out float pdf, out VisibilityTester visibility)
    {
        wi = normalize(-direction);
        pdf = 1.0;

        visibility.from = interaction;
        visibility.dir = -direction;
        visibility.dist = Infinity;
        
        return color.rgb * intensity;
    }
};

struct RectLight
{
    float3 color;
    float intensity;

    float3 Li(float3 p, out float3 wi)
    {
        return 0.f;
    }
    
    bool IsDelta()
    {
        return false;
    }
    
    float3 SampleLi(SurfaceInteraction interaction, float2 u, out float3 wi, out float pdf, out VisibilityTester visibility)
    {
        return 0.f;
    }
};

// Shadow Map
// Texture2D<float4> Textures[] : register(s996);
// SamplerState Samplers[] : register(s997);
StructuredBuffer<uint> LightInfo : register(t988);
ByteAddressBuffer LightBuffer : register(t989);

struct Light
{
    PointLight point_light;
    SpotLight spot_light;
    DirectionalLight directional_light;
    RectLight rect_light;
    uint light_type;
    
    void Init(uint light_id_)
    {
        light_type = LightInfo[2 * light_id_];
        switch (light_type)
        {
            case POINT_LIGHT:
                point_light = LightBuffer.Load < PointLight > (LightInfo[2 * light_id_ + 1]);
                break;
            case SPOT_LIGHT:
                spot_light = LightBuffer.Load < SpotLight > (LightInfo[2 * light_id_ + 1]);
                break;
            case DIRECTIONAL_LIGHT:
                directional_light = LightBuffer.Load < DirectionalLight > (LightInfo[2 * light_id_ + 1]);
                break;
            case RECT_LIGHT:
                rect_light = LightBuffer.Load < RectLight > (LightInfo[2 * light_id_ + 1]);
                break;
        }
    }
    
    LightLiSample SampleLi(LightSampleContext ctx, float2 u)
    {
        switch (light_type)
        {
            case POINT_LIGHT:
                return point_light.SampleLi(ctx, u);
        }
        
        LightLiSample light_sample;
        return light_sample;
    }

    float PDF_Li(LightSampleContext ctx, float3 wi)
    {
        switch (light_type)
        {
            case POINT_LIGHT:
                return point_light.PDF_Li(ctx, wi);
        }
        
        return 0;
    }
    
    float3 Li(float3 p, out float3 wi)
    {
        switch (light_type)
        {
            case POINT_LIGHT:
                return point_light.Li(p, wi);
            case SPOT_LIGHT:
                return spot_light.Li(p, wi);
            case DIRECTIONAL_LIGHT:
                return directional_light.Li(p, wi);
            case RECT_LIGHT:
                return rect_light.Li(p, wi);
        }
        return 0.f;
    }
    
    bool IsDelta()
    {
        switch (light_type)
        {
            case POINT_LIGHT:
                return point_light.IsDelta();
            case SPOT_LIGHT:
                return spot_light.IsDelta();
            case DIRECTIONAL_LIGHT:
                return directional_light.IsDelta();
            case RECT_LIGHT:
                return rect_light.IsDelta();
        }
        return false;
    }
    
    float3 SampleLi(SurfaceInteraction interaction, float2 u, out float3 wi, out float pdf, out VisibilityTester visibility)
    {
        switch (light_type)
        {
            case POINT_LIGHT:
                return point_light.SampleLi(interaction, u, wi, pdf, visibility);
            case SPOT_LIGHT:
                return spot_light.SampleLi(interaction, u, wi, pdf, visibility);
            case DIRECTIONAL_LIGHT:
                return directional_light.SampleLi(interaction, u, wi, pdf, visibility);
            case RECT_LIGHT:
                return rect_light.SampleLi(interaction, u, wi, pdf, visibility);
        }
        return 0.f;
    }
};

#endif