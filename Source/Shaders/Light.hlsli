#ifndef __LIGHT_HLSL__
#define __LIGHT_HLSL__

#include "Common.hlsli"
#include "Constants.hlsli"
#include "Math.hlsli"

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
    
    float3 SampleLi(Interaction interaction, float2 u, out float3 wi, out float pdf)
    {
        wi = normalize(-direction);
        pdf = 1.0;

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
    
    float3 SampleLi(Interaction interaction, float2 u, out float3 wi, out float pdf)
    {
        wi = normalize(position - interaction.position);
        pdf = 1.0;

        float d = length(position - interaction.position);
        float Fatt = 1.0 / (constant + linear_ * d + quadratic * d * d);
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
    
    float3 SampleLi(Interaction interaction, float2 u, out float3 wi, out float pdf)
    {
        wi = normalize(position - interaction.position);
        pdf = 1.0;

        float3 L = normalize(position - interaction.position);
        float NoL = max(0.0, dot(interaction.normal, L));
        float theta = dot(L, normalize(-direction));
        float epsilon = cut_off - outer_cut_off;
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

#endif