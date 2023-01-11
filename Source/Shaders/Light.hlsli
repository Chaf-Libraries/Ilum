#ifndef LIGHT_HLSLI
#define LIGHT_HLSLI

#include "Interaction.hlsli"
#include "Math.hlsli"

#define POINT_LIGHT 0
#define SPOT_LIGHT 1
#define DIRECTIONAL_LIGHT 2
#define RECT_LIGHT 3

struct PointLight
{
    float3 color;
    float intensity;
    float3 position;
    float radius;
    float range;

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
                point_light = LightBuffer.Load<PointLight>(LightInfo[2 * light_id_ + 1]);
                break;
            case SPOT_LIGHT:
                spot_light = LightBuffer.Load<SpotLight>(LightInfo[2 * light_id_ + 1]);
                break;
            case DIRECTIONAL_LIGHT:
                directional_light = LightBuffer.Load<DirectionalLight>(LightInfo[2 * light_id_ + 1]);
                break;
            case RECT_LIGHT:
                rect_light = LightBuffer.Load<RectLight>(LightInfo[2 * light_id_ + 1]);
                break;
        }
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