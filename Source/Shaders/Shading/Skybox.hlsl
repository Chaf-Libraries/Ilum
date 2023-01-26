#include "Common.hlsli"

ConstantBuffer<View> ViewBuffer;
TextureCube Skybox;
SamplerState SkyboxSampler;

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
};

struct PSOutput
{
    float4 Color : SV_Target0;
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
    output.Pos = mul(ViewBuffer.view_projection_matrix, float4(output.UVW + ViewBuffer.position, 1.0)).xyww;
    return output;
}

PSOutput PSmain(PSInput input)
{
    PSOutput output;
    output.Color = float4(Skybox.Sample(SkyboxSampler, input.UVW).rgb, 1.0);
    return output;
}