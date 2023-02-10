#include "Material/BSDF/DiffuseBSDF.hlsli"
#include "Material/BSDF/BlendBSDF.hlsli"
#include "Common.hlsli"

RWTexture2D<float4> Output;

[numthreads(32, 32, 1)]
void CSmain(CSParam param)
{
    DiffuseBSDF diffuse1;
    DiffuseBSDF diffuse2;
    diffuse1.Init(1.f);
    diffuse1.Init(0.f);
    BlendBSDF< DiffuseBSDF, DiffuseBSDF > blend_bsdf1;
    BlendBSDF< DiffuseBSDF, BlendBSDF< DiffuseBSDF, DiffuseBSDF > > blend_bsdf;
    blend_bsdf1.Init(diffuse1, diffuse2, 0.2f);
    blend_bsdf.Init(diffuse1, blend_bsdf1, 0.2f);
    Output[param.DispatchThreadID.xy] = float4(blend_bsdf.Eval(0, 1, TransportMode_Radiance), 1.0);
}