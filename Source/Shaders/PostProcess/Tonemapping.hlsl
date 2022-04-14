#define TONEMAP_UNCHARTED
#include "../Tonemapper.hlsli"

#define LOCAL_SIZE 8

Texture2D InImage : register(t0);
SamplerState texSampler : register(s0);
RWTexture2D<float4> OutImage : register(u1);

[[vk::push_constant]]
struct
{
    float brightness;
    float contrast;
    float saturation;
    float vignette;
    float avgLum;
    int autoExposure;
    float Ywhite; // Burning white
    float key; // Log-average luminance
} push_constants;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

// http://www.thetenthplanet.de/archives/5367
// Apply dithering to hide banding artifacts.
float3 Dither(float3 linear_color, float3 noise, float quant)
{
    float3 c0 = floor(LinearTosRGB(linear_color) / quant) * quant;
    float3 c1 = c0 + quant;
    float3 discr = lerp(sRGBToLinear(c0), sRGBToLinear(c1), noise);
    return lerp(c0, c1, float3(1.0, 1.0, 1.0) - step(linear_color, discr));
}

// http://user.ceng.metu.edu.tr/~akyuz/files/hdrgpu.pdf
static const float3x3 RGB2XYZ = float3x3(0.4124564, 0.3575761, 0.1804375, 0.2126729, 0.7151522, 0.0721750, 0.0193339, 0.1191920, 0.9503041);
float Luminance(float3 color)
{
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f)); //color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;
}

float3 ToneExposure(float3 RGB, float logAvgLum)
{
    float3 XYZ = mul(RGB2XYZ, RGB);
    float Y = (push_constants.key / logAvgLum) * XYZ.y;
    float Yd = (Y * (1.0 + Y / (push_constants.Ywhite * push_constants.Ywhite))) / (1.0 + Y);
    return RGB / XYZ.y * Yd;
}

float3 ToneLocalExposure(float3 RGB, float logAvgLum, float2 uvCoords)
{
    float3 XYZ = mul(RGB2XYZ, RGB);
    float Y = (push_constants.key / logAvgLum) * XYZ.y;
    float La; // local adaptation luminance
    float factor = push_constants.key / logAvgLum;
    float epsilon = 0.05, phi = 2.0;
    float scale[7] = { 1, 2, 4, 8, 16, 32, 64 };
    for (int i = 0; i < 7; ++i)
    {
        float v1 = Luminance(InImage.SampleLevel(texSampler, uvCoords, i).rgb) * factor;
        float v2 = Luminance(InImage.SampleLevel(texSampler, uvCoords, i + 1).rgb) * factor;
        if (abs(v1 - v2) / ((push_constants.key * pow(2, phi) / (scale[i] * scale[i])) + v1) > epsilon)
        {
            La = v1;
            break;
        }
        else
            La = v2;
    }
    float Yd = Y / (1.0 + La);
    return RGB / XYZ.y * Yd;
}

uint3 pcg3d(uint3 v)
{
    v = v * 1664525u + uint3(1013904223u, 1013904223u, 1013904223u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> uint3(16u, 16u, 16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

[numthreads(LOCAL_SIZE, LOCAL_SIZE, 1)]
void main(CSParam param)
{
    uint2 TexSize;
    InImage.GetDimensions(TexSize.x, TexSize.y);
    float2 uvCoords = (float2(param.DispatchThreadID.xy) + float2(0.5, 0.5)) / float2(TexSize);
    float4 hdr = InImage.SampleLevel(texSampler, uvCoords, 0).rgba;
    
    if (((push_constants.autoExposure >> 0) & 1) == 1)
    {
        float4 avg = InImage.SampleLevel(texSampler, float2(0.5, 0.5), 20).rgba; // Get the average value of the image
        float avgLum2 = Luminance(avg.rgb); // Find the luminance
        if (((push_constants.autoExposure >> 1) & 1) == 1)
        {
            hdr.rgb = ToneLocalExposure(hdr.rgb, avgLum2, uvCoords); // Adjust exposure
        }
        else
        {
            hdr.rgb = ToneExposure(hdr.rgb, avgLum2); // Adjust exposure
        }
    }

    // Tonemap + Linear to sRgb
    float3 color = ToneMap(hdr.rgb, push_constants.avgLum);

    // Remove banding
    uint3 r = pcg3d(uint3(param.DispatchThreadID.xy, 0));
    float3 noise = asfloat(0x3f800000 | (r >> 9)) - 1.0f;
    color = Dither(sRGBToLinear(color), noise, 1. / 255.);

    //contrast
    color = clamp(lerp(float3(0.5, 0.5, 0.5), color, push_constants.contrast), 0, 1);
    // brighness
    float inv_brightness = 1.0 / push_constants.brightness;
    color = pow(color, float3(inv_brightness, inv_brightness, inv_brightness));
    // saturation
    float s = dot(color, float3(0.299, 0.587, 0.114));
    float3 i = float3(s, s, s);
    color = lerp(i, color, push_constants.saturation);
    // vignette
    float2 uv = (uvCoords - 0.5) * 2.0;
    color *= 1.0 - dot(uv, uv) * push_constants.vignette;

    OutImage[int2(param.DispatchThreadID.xy)] = float4(color, hdr.a);
}