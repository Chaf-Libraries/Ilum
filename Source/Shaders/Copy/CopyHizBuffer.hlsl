#define LOCAL_SIZE 32

Texture2D GBuffer : register(t0);
SamplerState GBufferSampler : register(s0);
RWTexture2D<float> HizBuffer : register(u1);

[[vk::push_constant]]
struct
{
    uint2 extent;
} push_constants;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

[numthreads(LOCAL_SIZE, LOCAL_SIZE, 1)]
void main(CSParam param)
{
    if(param.DispatchThreadID.x > push_constants.extent.x || param.DispatchThreadID.y > push_constants.extent.y)
    {
        return;
    }
    
    float2 uv = (float2(param.DispatchThreadID.xy) + float2(0.5, 0.5)) / float2(push_constants.extent);
    HizBuffer[int2(param.DispatchThreadID.xy)] = GBuffer.SampleLevel(GBufferSampler, uv, 0.0).a;
}