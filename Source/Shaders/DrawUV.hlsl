#include "Common.hlsli"

ConstantBuffer<ViewInfo> View : register(b0);

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float3 Color : COLOR0;
};

struct PSInput
{
    float3 Color : COLOR0;
};

[[vk::push_constant]]
struct
{
    float a;
    float b;
} push_constants;

static float3 positions[] =
{
    float3(-0.5f, -0.5f, -0.5f),
   float3(0.5f, -0.5f, -0.5f),
   float3(0.5f, 0.5f, -0.5f),
   float3(0.5f, 0.5f, -0.5f),
   float3(-0.5f, 0.5f, -0.5f),
   float3(-0.5f, -0.5f, -0.5f),
   float3(-0.5f, -0.5f, 0.5f),
   float3(0.5f, -0.5f, 0.5f),
   float3(0.5f, 0.5f, 0.5f),
   float3(0.5f, 0.5f, 0.5f),
   float3(-0.5f, 0.5f, 0.5f),
   float3(-0.5f, -0.5f, 0.5f),
   float3(-0.5f, 0.5f, 0.5f),
   float3(-0.5f, 0.5f, -0.5f),
   float3(-0.5f, -0.5f, -0.5f),
   float3(-0.5f, -0.5f, -0.5f),
   float3(-0.5f, -0.5f, 0.5f),
   float3(-0.5f, 0.5f, 0.5f),
   float3(0.5f, 0.5f, 0.5f),
   float3(0.5f, 0.5f, -0.5f),
   float3(0.5f, -0.5f, -0.5f),
   float3(0.5f, -0.5f, -0.5f),
   float3(0.5f, -0.5f, 0.5f),
   float3(0.5f, 0.5f, 0.5f),
   float3(-0.5f, -0.5f, -0.5f),
   float3(0.5f, -0.5f, -0.5f),
   float3(0.5f, -0.5f, 0.5f),
   float3(0.5f, -0.5f, 0.5f),
   float3(-0.5f, -0.5f, 0.5f),
   float3(-0.5f, -0.5f, -0.5f),
   float3(-0.5f, 0.5f, -0.5f),
   float3(0.5f, 0.5f, -0.5f),
   float3(0.5f, 0.5f, 0.5f),
   float3(0.5f, 0.5f, 0.5f),
   float3(-0.5f, 0.5f, 0.5f),
   float3(-0.5f, 0.5f, -0.5f)
};

static float3 colors[3] =
{
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0)
};

VSOutput VSmain(uint VertexIndex : SV_VertexID)
{
    VSOutput output = (VSOutput) 0;
    output.Pos = mul(View.view_projection_matrix, float4(positions[VertexIndex] + float3(1, 10*sin(push_constants.a*0.01), 1), 1.0));
    //output.Pos = float4(positions[VertexIndex], 0.0, 1.0);
    output.Color = clamp(positions[VertexIndex], 0.f, 1.f);
    return output;
}

float4 PSmain(PSInput input) : SV_TARGET
{
    return float4(input.Color+push_constants.b, 1);
    //return 1.f;
}

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

//RWStructuredBuffer<float2> VarX : register(u0);
//RWStructuredBuffer<float2> VarY : register(u1);
//struct UINT2
//{
//    uint x;
//    uint y;
//};
//
//struct FLOAT
//{
//    float data;
//};
//ConstantBuffer<FLOAT> Time : register(b3);
//
//RWTexture2D<float4> Tex : register(u1);
//ConstantBuffer<UINT2> TexSize : register(b2);

//RWTexture2D<float4> Tex : register(u1);
//RWStructuredBuffer<float2> VarY : register(u1);

//[numthreads(8, 8, 1)]
//void CSmain(uint2 DispatchID : SV_DispatchThreadID)
//{
//    
//    VarX[DispatchID.x] = float2(TexSize.x, TexSize.y);
//    //VarY[DispatchID.x] += 2;
//    Tex[DispatchID.xy] = float4(float2(DispatchID.xy) / float2(TexSize.x, TexSize.y), 0.0, 1.0);
//    
//    //VarY[DispatchID.x] = DispatchID.xy;
//}

// float opSmoothUnion(float d1, float d2, float k)
// {
//     float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
//     return lerp(d2, d1, h) - k * h * (1.0 - h);
// }
// 
// float sdSphere(float3 p, float s)
// {
//     return length(p) - s;
// }
// 
// float map(float3 p)
// {
//     float d = 2.0;
//     for (int i = 0; i < 16; i++)
//     {
//         float fi = float(i);
//         float time = Time.data.x * (1.0 / (fi * 412.531 + 0.513) - 0.5) * 2.0;
//         d = opSmoothUnion(
//             sdSphere(p + sin(time + fi * float3(52.5126, 64.62744, 632.25)) * float3(2.0, 2.0, 0.8), lerp(0.5, 1.0, 1.0 / (fi * 412.531 + 0.5124))),
// 			d,
// 			0.4
// 		);
//     }
//     return d;
// }
// 
// float3 calcNormal(in float3 p)
// {
//     const float h = 1e-5; // or some other value
//     const float2 k = float2(1, -1);
//     return normalize(k.xyy * map(p + k.xyy * h) +
//                       k.yyx * map(p + k.yyx * h) +
//                       k.yxy * map(p + k.yxy * h) +
//                       k.xxx * map(p + k.xxx * h));
// }
// 
// [numthreads(8, 8, 1)]
// void CSmain(uint2 DispatchID : SV_DispatchThreadID)
// {
//     float2 uv = float2(DispatchID.xy) / float2(TexSize.x, TexSize.y);
//     
//     // screen size is 6m x 6m
//     float3 rayOri = float3((uv - 0.5) * float2(TexSize.x / TexSize.y, 1.0) * 6.0, 3.0);
//     float3 rayDir = float3(0.0, 0.0, -1.0);
// 	
//     float depth = 0.0;
//     float3 p;
// 	
//     for (int i = 0; i < 64; i++)
//     {
//         p = rayOri + rayDir * depth;
//         float dist = map(p);
//         depth += dist;
//         if (dist < 1e-6)
//         {
//             break;
//         }
//     }
// 	
//     depth = min(6.0, depth);
//     float3 n = calcNormal(p);
//     float b = max(0.0, dot(n, float3(0.577, 0.577, 0.577)));
//     
//     //float3 col = (0.5 + 0.5 * cos((b + Time.data * 3.0)    )) * (0.85 + b * 0.35);
//     
//     float3 col = (0.5 + 0.5 * cos((b + Time.data.x * 3.0) + float3(uv.x, uv.y, uv.x) * 2.0 + float3(0, 2, 4))) * (0.85 + b * 0.35);
//     col *= exp(-depth * 0.15);
// 	
//     // maximum thickness is 2m in alpha channel
//     Tex[DispatchID.xy] = float4(col, 1.0 - (depth - 0.5) / 2.0);
//     VarX[DispatchID.x] += 1;
// }
