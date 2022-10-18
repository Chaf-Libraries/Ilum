#ifndef LIGHT_SOURCE_HLSLI
#define LIGHT_SOURCE_HLSLI

static const uint LightType_Point = 0;
static const uint LightType_Spot = 1;
static const uint LightType_Directional = 2;
static const uint LightType_Area = 3;
static const uint LightType_Unknown = 4;

struct Light
{
    uint type;
    float3 color;
    float intensity;
    float3 position;
    float range; // Point
    float radius; // Point/Spot
    float cut_off; // Spot
    float outer_cut_off; // Spot
    float3 direction; // Spot/Directional
    
    float3 Li(float3 p, out float3 wi)
    {
        if (type == LightType_Point)
        {
            wi = normalize(position - p);
            float d = length(position - p);
            float attenuation = max(min(1.0 - pow(d / range, 4.0), 1.0), 0.0) / (d * d);
            return color.rgb;// * intensity * attenuation;
        }
        return 1.f;
        switch (type)
        {
            case LightType_Point:
            {
                    wi = normalize(position - p);
                    float d = length(position - p);
                    float attenuation = max(min(1.0 - pow(d / range, 4.0), 1.0), 0.0) / (d * d);
                    return color.rgb * intensity * attenuation;
                }

            case LightType_Spot:
                {
                    wi = normalize(position - p);
                    float light_angle_scale = 1.0 / max(0.001, cos(cut_off) - cos(outer_cut_off));
                    float light_angle_offset = -cos(outer_cut_off) * light_angle_scale;
                    float cd = max(dot(direction, wi), 0.0);
                    float angular_attenuation = saturate(cd * light_angle_scale + light_angle_offset);
                    return color.rgb * intensity * angular_attenuation * angular_attenuation;
                }
            case LightType_Directional:
{
                    wi = normalize(-direction);
                    return color.rgb * intensity;
                }
            case LightType_Area:
                break;
        }

        return 0.f;
    }
};

StructuredBuffer<Light> LightBuffer;

#endif