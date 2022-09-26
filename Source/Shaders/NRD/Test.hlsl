RWTexture2D<float4> OutputImage;
Texture2D<float> InputImage;
SamplerState _sampler;

cbuffer Constant
{
    uint2 TexSize;
};

[numthreads(8, 8, 1)]
void MainCS(int2 DispatchID : SV_DispatchThreadID)
{
    OutputImage[DispatchID] = float4(InputImage.SampleLevel(_sampler, float2(DispatchID) / float2(TexSize), 0.f).rrr, 1.f);

}