RWTexture2D<float4> OutputImage;
Texture2D<float4> InputImage;
SamplerState _sampler;

float3 _NRD_DecodeUnitVector(float2 p, const bool bSigned = false, const bool bNormalize = true)
{
    p = bSigned ? p : (p * 2.0 - 1.0);

    // https://twitter.com/Stubbesaurus/status/937994790553227264
    float3 n = float3(p.xy, 1.0 - abs(p.x) - abs(p.y));
    float t = saturate(-n.z);
    n.xy += n.xy >= 0.0 ? -t : t;

    return bNormalize ? normalize(n) : n;
}
#define NRD_USE_OCT_NORMAL_ENCODING 1
#define NRD_USE_MATERIAL_ID 1
float4 NRD_FrontEnd_UnpackNormalAndRoughness(float4 p, out float materialID)
{
    materialID = 0;

    float4 r;
#if( NRD_USE_OCT_NORMAL_ENCODING == 1 )
        r.xyz = _NRD_DecodeUnitVector( p.xy, false, false );
        r.w = p.z;

#if( NRD_USE_MATERIAL_ID == 1 )
            materialID = p.w;
#endif
#else
    r.xyz = p.xyz * 2.0 - 1.0;
    r.w = p.w;
#endif

    r.xyz = normalize(r.xyz);

#if( NRD_USE_SQRT_LINEAR_ROUGHNESS == 1 )
        r.w *= r.w;
#endif

    return r;
}

// IN_NORMAL_ROUGHNESS => X
float4 NRD_FrontEnd_UnpackNormalAndRoughness(float4 p)
{
    float unused;

    return NRD_FrontEnd_UnpackNormalAndRoughness(p, unused);
}

float4 UnpackNormalAndRoughness(float4 p, bool isNormalized = true)
{
    p.xyz = p.xyz * 2.0 - 1.0;

    if (isNormalized)
        p.xyz = normalize(p.xyz);

    return p;
}

float3 RotateVector(float4x4 m, float3 v)
{
    return mul((float3x3) m, v);
}

[numthreads(8, 8, 1)]
void MainCS(int2 DispatchID : SV_DispatchThreadID)
{
    float4x4 identity = float4x4(
    float4(1, 0, 0, 0),
    float4(0, 1, 0, 0),
    float4(0, 0, 1, 0),
    float4(0, 0, 0, 1)
);

    float3 prevNflat = UnpackNormalAndRoughness(InputImage.SampleLevel(_sampler, float2(DispatchID) / float2(1280, 720), 0)).xyz;
    prevNflat = RotateVector(identity, prevNflat);
    
    //OutputImage[DispatchID] = float4(InputImage.SampleLevel(_sampler, float2(DispatchID) / float2(1280, 720), 0.f).x, 0.f, 0.f, 1.f);
    OutputImage[DispatchID] = float4(prevNflat, 1.f);
//    float4(NRD_FrontEnd_UnpackNormalAndRoughness(InputImage.SampleLevel(_sampler, float2(DispatchID) / float2(1280, 720), 0)).xyz, 1.0);

}