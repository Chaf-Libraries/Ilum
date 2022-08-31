//struct VSOutput
//{
//    float4 Pos : SV_POSITION;
//};
//
//static float2 positions[3] =
//{
//    float2(0.0, -0.5),
//    float2(0.5, 0.5),
//    float2(-0.5, 0.5)
//};
//
//static float3 colors[3] =
//{
//    float3(1.0, 0.0, 0.0),
//    float3(0.0, 1.0, 0.0),
//    float3(0.0, 0.0, 1.0)
//};
//
//struct ConstantData
//{
//    float4 data;
//};
//
//ConstantBuffer<ConstantData> ConstantColor : register(b0, space0);
//
//VSOutput VSmain(uint VertexIndex : SV_VertexID)
//{
//    VSOutput output = (VSOutput) 0;
//    output.Pos = float4(positions[VertexIndex], 0.0, 1.0);
//    return output;
//}
//
//float4 PSmain() : SV_TARGET
//{
//    return float4(ConstantColor.data.xyz, 1.f);
//}

//struct UINT2
//{
//    uint x;
//    uint y;
//};
//
//
//RWTexture2D<float4> Result : register(u0);
//ConstantBuffer<UINT2> TexSize : register(b1);
//
//[numthreads(8, 8, 1)]
//void CSmain(uint2 DispatchID : SV_DispatchThreadID)
//{
//    if (DispatchID.x >= TexSize.x || DispatchID.y >= TexSize.y)
//    {
//        return;
//    }
//    Result[DispatchID.xy] = float4(float2(DispatchID.xy) / float2(TexSize.x, TexSize.y), 0.0, 1.0);
//}

RWStructuredBuffer<float2> VarX : register(u0);
//RWStructuredBuffer<float2> VarY : register(u1);

[numthreads(8, 8, 1)]
void CSmain(uint2 DispatchID : SV_DispatchThreadID)
{
    VarX[DispatchID.x] = DispatchID.xy;
    //VarY[DispatchID.x] = DispatchID.xy;
}