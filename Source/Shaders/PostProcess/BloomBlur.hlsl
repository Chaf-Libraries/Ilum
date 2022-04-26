RWTexture2D<float4> BloomBlurIn : register(u0);
RWTexture2D<float4> BloomBlurOut : register(u1);

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint GroupIndex : SV_GroupIndex;
    uint3 GroupID : SV_GroupID;
    uint3 GroupThreadID : SV_GroupThreadID;
};

static const float blur_weights[5] = { 0.2734375, 0.21875, 0.109375, 0.03125, 0.00390625 };

groupshared uint3 shared_cache[128];
groupshared float3 shared_tile[64];

float3 BlurPixels(float3 a, float3 b, float3 c, float3 d, float3 e, float3 f, float3 g, float3 h, float3 i)
{
    return blur_weights[0] * e + blur_weights[1] * (d + f) + blur_weights[2] * (c + g) + blur_weights[3] * (b + h) + blur_weights[4] * (a + i);
}

void Store2Pixels(uint index, float3 pixel1, float3 pixel2)
{
    shared_cache[index] = f32tof16(pixel1) | f32tof16(pixel2) << 16;
}

void Load2Pixels(uint index, out float3 pixel1, out float3 pixel2)
{
    pixel1 = float3(f16tof32(shared_cache[index]));
    pixel2 = float3(f16tof32(shared_cache[index] >> 16));
}

void Store1Pixels(uint index, float3 pixel)
{
    shared_cache[index] = asuint(pixel);
}

void Load1Pixels(uint index, out float3 pixel)
{
    pixel = asfloat(shared_cache[index]);
}

void BlurHorizontally(uint index, uint start)
{
    float3 pixels[10];
    Load2Pixels(start + 0, pixels[0], pixels[1]);
    Load2Pixels(start + 1, pixels[2], pixels[3]);
    Load2Pixels(start + 2, pixels[4], pixels[5]);
    Load2Pixels(start + 3, pixels[6], pixels[7]);
    Load2Pixels(start + 4, pixels[8], pixels[9]);

    Store1Pixels(index, BlurPixels(pixels[0], pixels[1], pixels[2], pixels[3], pixels[4], pixels[5], pixels[6], pixels[7], pixels[8]));
    Store1Pixels(index + 1, BlurPixels(pixels[1], pixels[2], pixels[3], pixels[4], pixels[5], pixels[6], pixels[7], pixels[8], pixels[9]));
}

float3 BlurVertically(uint start)
{
    float3 pixels[9];
    Load1Pixels(start, pixels[0]);
    Load1Pixels(start + 8, pixels[1]);
    Load1Pixels(start + 16, pixels[2]);
    Load1Pixels(start + 24, pixels[3]);
    Load1Pixels(start + 32, pixels[4]);
    Load1Pixels(start + 40, pixels[5]);
    Load1Pixels(start + 48, pixels[6]);
    Load1Pixels(start + 56, pixels[7]);
    Load1Pixels(start + 64, pixels[8]);
    
    return BlurPixels(pixels[0], pixels[1], pixels[2], pixels[3], pixels[4], pixels[5], pixels[6], pixels[7], pixels[8]);
}

[numthreads(8, 8, 1)]
void main(CSParam param)
{
    uint parity = param.DispatchThreadID.x | param.DispatchThreadID.y;  
        
    int2 GroupUL = (param.GroupID.xy << 3) - 4;
    int2 ThreadUL = (param.GroupThreadID.xy << 1) + GroupUL;
    
    int start = param.GroupThreadID.x + (param.GroupThreadID.y << 4);
    
    Store2Pixels(
        start + 0,
        BloomBlurIn.Load(uint3(ThreadUL + uint2(0, 0), 0)).rgb,
        BloomBlurIn.Load(uint3(ThreadUL + uint2(1, 0), 0)).rgb
    );
    
    Store2Pixels(
        start + 8,
        BloomBlurIn.Load(uint3(ThreadUL + uint2(0, 1), 0)).rgb,
        BloomBlurIn.Load(uint3(ThreadUL + uint2(1, 1), 0)).rgb
    );
    
    GroupMemoryBarrierWithGroupSync();
    
    uint row = param.GroupThreadID.y << 4;
    BlurHorizontally(row + (param.GroupThreadID.x << 1), row + param.GroupThreadID.x + (param.GroupThreadID.x & 4));
    
    GroupMemoryBarrierWithGroupSync();
    
    BloomBlurOut[param.DispatchThreadID.xy] = float4(BlurVertically((param.GroupThreadID.y << 3) + param.GroupThreadID.x), 1.0);
}