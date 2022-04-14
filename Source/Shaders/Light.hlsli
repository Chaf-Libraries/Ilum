#ifndef __LIGHT_HLSL__
#define __LIGHT_HLSL__

#include "Common.hlsli"
#include "Constants.hlsli"
#include "Math.hlsli"

struct VisibilityTester
{
    Interaction from;
    float3 dir;
    float dist;
};

// DIrectional Light Source
struct DirectionalLight
{
    float4 split_depth;
    float4x4 view_projection[4];
    float3 color;
    float intensity;
    float3 direction;

    int shadow_mode; // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS
    float filter_scale;
    int filter_sample;
    int sample_method; // 0 - Uniform, 1 - Poisson Disk
    float light_size;

    float3 position;
    
    float3 SampleLi(Interaction interaction, float2 u, out float3 wi, out float pdf, out VisibilityTester visibility)
    {
        wi = normalize(-direction);
        pdf = 1.0;

        visibility.from = interaction;
        visibility.dir = -direction;
        visibility.dist = Infinity;
        
        return color.rgb * intensity;
    }

    float PdfLi(Interaction interaction, float3 wi)
    {
        return 0.0;
    }

    float Power()
    {
        return Infinity;
    }
};

// Point Light Source
struct PointLight
{
    float3 color;
    float intensity;
    float3 position;
    float constant;
    float linear_;
    float quadratic;

    int shadow_mode; // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS
    float filter_scale;
    int filter_sample;
    int sample_method; // 0 - Uniform, 1 - Poisson Disk
    float light_size;
    
    float3 SampleLi(Interaction interaction, float2 u, out float3 wi, out float pdf, out VisibilityTester visibility)
    {
        wi = normalize(position - interaction.position);
        pdf = 1.0;

        float d = length(position - interaction.position);
        float Fatt = 1.0 / (constant + linear_ * d + quadratic * d * d);
        
        visibility.from = interaction;
        visibility.dir = normalize(position - interaction.position);
        visibility.dist = length(position - interaction.position);
        
        return color.rgb * intensity * Fatt;
    }
    
    float PdfLi(Interaction interaction, float3 wi)
    {
        return 0.0;
    }

    float Power()
    {
        return 4 * PI * intensity;
    }
};

// Spot Light Source
struct SpotLight
{
    float4x4 view_projection;
    float3 color;
    float intensity;
    float3 position;
    float cut_off;
    float3 direction;
    float outer_cut_off;

    int shadow_mode; // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS
    float filter_scale;
    int filter_sample;
    int sample_method; // 0 - Uniform, 1 - Poisson Disk
    float light_size;
    
    float3 SampleLi(Interaction interaction, float2 u, out float3 wi, out float pdf, out VisibilityTester visibility)
    {
        wi = normalize(position - interaction.position);
        pdf = 1.0;

        float3 L = normalize(position - interaction.position);
        float NoL = max(0.0, dot(interaction.normal, L));
        float theta = dot(L, normalize(-direction));
        float epsilon = cut_off - outer_cut_off;

        visibility.from = interaction;
        visibility.dir = normalize(position - interaction.position);
        visibility.dist = length(position - interaction.position);
        
        return color * intensity * clamp((theta - outer_cut_off) / epsilon, 0.0, 1.0);
    }

    float PdfLi(Interaction interaction, float3 wi)
    {
        return 0.0;
    }

    float Power()
    {
        return intensity * 2.0 * PI * (1.0 - 0.5 * (cut_off - outer_cut_off));
    }
};

static const uint LightFlag_None = 1;
static const uint LightFlag_DeltaPosition = 1 << 1;
static const uint LightFlag_DeltaDirection = 1 << 2;

static const uint LightType_None = 0;
static const uint LightType_Directional = 1;
static const uint LightType_Point = 2;
static const uint LightType_Spot = 3;

struct Light
{
    DirectionalLight directional_light;
    SpotLight spot_light;
    PointLight point_light;
    
    uint light_type;
    
    float3 SampleLi(Interaction interaction, float2 u, out float3 wi, out float pdf, out VisibilityTester visibility)
    {
        switch (light_type)
        {
            case LightType_Directional:
                return directional_light.SampleLi(interaction, u, wi, pdf, visibility);
            case LightType_Spot:
                return spot_light.SampleLi(interaction, u, wi, pdf, visibility);
            case LightType_Point:
                return point_light.SampleLi(interaction, u, wi, pdf, visibility);
        }
        
        wi = float3(0.0, 0.0, 0.0);
        pdf = 0.0;
        return float3(0.0, 0.0, 0.0);
    }
    
    float PdfLi(Interaction interaction, float3 wi)
    {
        switch (light_type)
        {
            case LightType_Directional:
                return directional_light.PdfLi(interaction, wi);
            case LightType_Spot:
                return spot_light.PdfLi(interaction, wi);
            case LightType_Point:
                return point_light.PdfLi(interaction, wi);
        }
        return 0.0;
    }

    float Power()
    {
        switch (light_type)
        {
            case LightType_Directional:
                return directional_light.Power();
            case LightType_Spot:
                return spot_light.Power();
            case LightType_Point:
                return point_light.Power();
        }
        return 0.0;
    }
};

uint GetLightFlag(Light light)
{
    switch (light.light_type)
    {
        case LightType_Directional:
            return LightFlag_DeltaDirection;
        case LightType_Point:
            return LightFlag_DeltaPosition;
        case LightType_Spot:
            return LightFlag_DeltaPosition;
    }
    return LightFlag_None;
}

bool IsDeltaLight(Light light)
{
    return GetLightFlag(light) & LightFlag_DeltaPosition ||
           GetLightFlag(light) & LightFlag_DeltaDirection;
}

#endif