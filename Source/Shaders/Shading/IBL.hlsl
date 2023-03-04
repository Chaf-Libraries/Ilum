#include "../Common.hlsli"
#include "../SphericalHarmonic.hlsli"

#define POSITIVE_X 0
#define NEGATIVE_X 1
#define POSITIVE_Y 2
#define NEGATIVE_Y 3
#define POSITIVE_Z 4
#define NEGATIVE_Z 5

RWTexture2DArray<float4> SHIntermediate;
TextureCube Skybox;
SamplerState SkyboxSampler;

groupshared SH9Color shared_sh_coeffs[8][8];
groupshared float shared_weights[8][8];

[numthreads(8, 8, 1)]
void CubemapSHProjection(CSParam param)
{
    
}
