Texture2D PrevImage : register(t0);
SamplerState LinearSampler : register(s0);
Texture2D InImage : register(t1);
SamplerState PointSampler : register(s1);
Texture2D GBuffer3 : register(t2);
RWTexture2D<float4> OutImage : register(u3);

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

float3 RGBToYCoCg(float3 RGB)
{
    float Y = dot(RGB, float3(1, 2, 1));
    float Co = dot(RGB, float3(2, 0, -2));
    float Cg = dot(RGB, float3(-1, 2, -1));
    
    float3 YCoCg = float3(Y, Co, Cg);
    return YCoCg;
}

float3 YCoCgToRGB(float3 YCoCg)
{
    float Y = YCoCg.x * 0.25;
    float Co = YCoCg.y * 0.25;
    float Cg = YCoCg.z * 0.25;
    
    float R = Y + Co - Cg;
    float G = Y + Cg;
    float B = Y - Co - Cg;
    
    float3 RGB = float3(R, G, B);
    return RGB;
}

float Luma4(float3 Color)
{
    return (Color.g * 2.0) + (Color.r + Color.b);
}

[numthreads(32, 32, 1)]
void main(CSParam param)
{
    uint2 extent;
    OutImage.GetDimensions(extent.x, extent.y);
    
    if (param.DispatchThreadID.x > extent.x || param.DispatchThreadID.y > extent.y)
    {
        return;
    }
    
    float2 pixel_size = 1.0 / float2(extent);
    
    float2 uv = (float2(param.DispatchThreadID.xy + float2(0.5, 0.5))) / float2(extent);
    
    float2 velocity = GBuffer3.SampleLevel(LinearSampler, uv, 0.0).ba;
    float2 prev_uv = uv - velocity;
    
    float3 prev_color = PrevImage.SampleLevel(LinearSampler, prev_uv, 0.0).rgb;
    float3 current_color = InImage.SampleLevel(PointSampler, uv, 0.0).rgb;
    
    float3 near0 = InImage.SampleLevel(PointSampler, uv + float2(1.0, 0.0) * pixel_size, 0.0).rgb;
    float3 near1 = InImage.SampleLevel(PointSampler, uv + float2(0.0, 1.0) * pixel_size, 0.0).rgb;
    float3 near2 = InImage.SampleLevel(PointSampler, uv + float2(-1.0, 0.0) * pixel_size, 0.0).rgb;
    float3 near3 = InImage.SampleLevel(PointSampler, uv + float2(0.0, -1.0) * pixel_size, 0.0).rgb;
    
    float3 min_ = min(current_color, min(near0, min(near1, min(near2, near3))));
    float3 max_ = max(current_color, max(near0, max(near1, max(near2, near3))));
    
    prev_color = clamp(prev_color, min_, max_);
    
    OutImage[param.DispatchThreadID.xy] = float4(lerp(current_color, prev_color, 0.9), 1.0);
    
}