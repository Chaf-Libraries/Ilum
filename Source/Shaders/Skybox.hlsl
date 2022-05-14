#include "ShaderInterop.hpp"

ConstantBuffer<Camera> camera : register(b0);
Texture2D DepthStencil : register(t1);
TextureCube Skybox : register(t2);
SamplerState SkyboxSampler : register(s3);

struct VSInput
{
    uint VertexID : SV_VertexID;
};

struct VSOutput
{
    float4 Pos : SV_Position;
    float3 UVW : POSITION0;
};

struct PSInput
{
    float3 UVW : POSITION0;
    float4 Pos : SV_Position;
};

struct PSOutput
{
    float4 Color : SV_Target0;
    float Depth : SV_Depth;
};

// A cube
static const float3 vertices[] =
{
    { -1.0, -1.0, -1.0 }, // bottom-left
    { 1.0, 1.0, -1.0 }, // top-right
    { 1.0, -1.0, -1.0 }, // bottom-right
    { 1.0, 1.0, -1.0 }, // top-right
    { -1.0, -1.0, -1.0 }, // bottom-left
    { -1.0, 1.0, -1.0 }, // top-left
	                                  // front face
    { -1.0, -1.0, 1.0 }, // bottom-left
    { 1.0, -1.0, 1.0 }, // bottom-right
    { 1.0, 1.0, 1.0 }, // top-right
    { 1.0, 1.0, 1.0 }, // top-right
    { -1.0, 1.0, 1.0 }, // top-left
    { -1.0, -1.0, 1.0 }, // bottom-left
	                                  // left face
    { -1.0, 1.0, 1.0 }, // top-right
    { -1.0, 1.0, -1.0 }, // top-left
    { -1.0, -1.0, -1.0 }, // bottom-left
    { -1.0, -1.0, -1.0 }, // bottom-left
    { -1.0, -1.0, 1.0 }, // bottom-right
    { -1.0, 1.0, 1.0 }, // top-right
	                                  // right face
    { 1.0, 1.0, 1.0 }, // top-left
    { 1.0, -1.0, -1.0 }, // bottom-right
    { 1.0, 1.0, -1.0 }, // top-right
    { 1.0, -1.0, -1.0 }, // bottom-right
    { 1.0, 1.0, 1.0 }, // top-left
    { 1.0, -1.0, 1.0 }, // bottom-left
	                                  // bottom face
    { -1.0, -1.0, -1.0 }, // top-right
    { 1.0, -1.0, -1.0 }, // top-left
    { 1.0, -1.0, 1.0 }, // bottom-left
    { 1.0, -1.0, 1.0 }, // bottom-left
    { -1.0, -1.0, 1.0 }, // bottom-right
    { -1.0, -1.0, -1.0 }, // top-right
	                                  // top face
    { -1.0, 1.0, -1.0 }, // top-left
    { 1.0, 1.0, 1.0 }, // bottom-right
    { 1.0, 1.0, -1.0 }, // top-right
    { 1.0, 1.0, 1.0 }, // bottom-right
    { -1.0, 1.0, -1.0 }, // top-left
    { -1.0, 1.0, 1.0 }, // bottom-left
};

VSOutput VSmain(VSInput input)
{
    VSOutput output;
    output.UVW = vertices[input.VertexID];
    output.Pos = mul(camera.view_projection, float4(output.UVW + camera.position, 1.0)).xyww;
    return output;
}

PSOutput PSmain(PSInput input)
{
    PSOutput output;
    if (DepthStencil.Load(int3(input.Pos.xy, 0.0)).r < 1.0)
    {
        discard;
    }
    output.Color = float4(Skybox.Sample(SkyboxSampler, input.UVW.xyz).rgb, 1.0);
    return output;
}