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
    float radius;

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
    
    bool IsDelta()
    {
        if (radius > 0)
        {
            return false;
        }
        return true;
    }
    
    float Area()
    {
        return 2.0 * PI * radius;
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
    float radius;
    
    float3 SampleLi(Interaction interaction, float2 u, out float3 wi, out float pdf, out VisibilityTester visibility)
    {
        if (radius == 0.0)
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
        else
        {
            float3 p = radius * UniformSampleSphere(u) + position;
                        
            pdf = 1.0 / (2 * PI * radius * radius);
            
            wi = normalize(p - interaction.position);

            float d = length(p - interaction.position);
            float Fatt = 1.0 / (constant + linear_ * d + quadratic * d * d);
        
            visibility.from = interaction;
            visibility.dir = normalize(p - interaction.position);
            visibility.dist = length(p - interaction.position);
        
            return color.rgb * intensity * Fatt / (2 * PI * radius * radius);
        }
        
        return float3(0.0, 0.0, 0.0);
    }
    
    float PdfLi(Interaction interaction, float3 wi)
    {
        return 0.0;
    }

    float Power()
    {
        return 4 * PI * intensity;
    }
    
    bool IsDelta()
    {
        if (radius > 0)
        {
            return false;
        }
        return true;
    }
    
    float Area()
    {
        return 2.0 * PI * radius;
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
    float radius;
    
    float3 SampleLi(Interaction interaction, float2 u, out float3 wi, out float pdf, out VisibilityTester visibility)
    {
        if (radius == 0.0)
        {
            wi = normalize(position - interaction.position);
            pdf = 1.0;

            float3 L = normalize(position - interaction.position);
            float theta = dot(L, normalize(-direction));
            float epsilon = cut_off - outer_cut_off;

            visibility.from = interaction;
            visibility.dir = normalize(position - interaction.position);
            visibility.dist = length(position - interaction.position);
        
            return color * intensity * clamp((theta - outer_cut_off) / epsilon, 0.0, 1.0);
        }
        else
        {
            float3 p = radius * UniformSampleSphere(u) + position;
                        
            pdf = 1.0 / (2 * PI * radius * radius);
            
            wi = normalize(p - interaction.position);

            float3 L = normalize(p - interaction.position);
            float theta = dot(L, normalize(-direction));
            float epsilon = cut_off - outer_cut_off;

            visibility.from = interaction;
            visibility.dir = normalize(p - interaction.position);
            visibility.dist = length(p - interaction.position);
        
            return color * intensity * clamp((theta - outer_cut_off) / epsilon, 0.0, 1.0) / (2 * PI * radius * radius);
        }
    }

    float PdfLi(Interaction interaction, float3 wi)
    {
        return 0.0;
    }

    float Power()
    {
        return intensity * 2.0 * PI * (1.0 - 0.5 * (cut_off - outer_cut_off));
    }
    
    bool IsDelta()
    {
        if (radius > 0)
        {
            return false;
        }
        return true;
    }
    
    float Area()
    {
        return 2.0 * PI * radius;
    }
};

static const uint AreaLightType_Rectangle = 0;
static const uint AreaLightType_Ellipse = 1;

// Area Light Source
struct AreaLight
{
    float3 color;
    float intensity;

    float4 corners[4];
    
    uint shape;
    
    uint tex_id;
    
    float Area()
    {
        if (shape == AreaLightType_Rectangle)
        {
            float a = length(corners[0] - corners[1]);
            float b = length(corners[1] - corners[2]);
            return a * b;
        }
        else if (shape == AreaLightType_Ellipse)
        {
            float a = length(corners[0] - corners[1]) * 0.5;
            float b = length(corners[1] - corners[2]) * 0.5;
            return PI * a * b;
        }
        return 0.0;
    }
    
    float3 SampleLi(Interaction interaction, float2 u, out float3 wi, out float pdf, out VisibilityTester visibility)
    {
        float3 n;
        float3 p;
        
        float3 x_axis = (corners[1] - corners[0]).xyz;
        float3 y_axis = (corners[3] - corners[0]).xyz;
        
        // Sample Shape
        if (shape == AreaLightType_Rectangle)
        {
            float2 pd = u;
            p = corners[0].xyz + x_axis * pd.x + y_axis * pd.y;
            pdf = 1.0 / Area();
        }
        else if (shape == AreaLightType_Ellipse)
        {
            float2 pd = SampleConcentricDisk(u);
            p = corners[0].xyz + x_axis * pd.x + y_axis * pd.y + 0.5 * (corners[2] - corners[0]).xyz;
            pdf = 1.0 / Area();
        }
                
        wi = p - interaction.position;
        if (IsBlack(wi))
        {
            pdf = 0.0;
        }
        else
        {
            wi = normalize(wi);
            pdf *= dot(interaction.position - p, interaction.position - p) / abs(dot(interaction.ffnormal, wi));
            if (isinf(pdf))
            {
                pdf = 0.0;
            }
        }
        
        visibility.from = interaction;
        visibility.dir = normalize(p - interaction.position);
        visibility.dist = length(p - interaction.position);

        return color * intensity;
    }

    float PdfLi(Interaction interaction, float3 wi)
    {
        float3 normal = normalize(cross(corners[1].xyz - corners[0].xyz, corners[2].xyz - corners[0].xyz));
        
        // Check Intersection
        RayDesc ray = SpawnRay(interaction, wi);
        if (dot(ray.Direction, normal) == 0.0)
        {
            return 0.0;
        }

        float t = abs(dot(interaction.position - ray.Origin, normal) / (ray.Direction, normal));
        float3 p = ray.Origin + t * ray.Direction;
        
        float3 v = p - corners[0].xyz;
        
        if (dot(v, corners[1].xyz - corners[0].xyz) < 0.0 || dot(v, corners[3].xyz - corners[0].xyz) < 0.0)
        {
            return 0.0;
        }
        
        float pdf = dot(interaction.position - p, interaction.position - p) / abs(dot(interaction.ffnormal, wi)) * Area();
        
        if (isinf(pdf))
        {
            return 0.0;
        }
        
        return pdf;
    }
    
    bool IsDelta()
    {
        return true;
    }

    float Power()
    {
        return intensity * Area() * PI;
    }
};

#endif