struct VSOutput
{
    float4 Pos : SV_POSITION;
};

static float2 positions[3] =
{
    float2(0.0, -0.5),
    float2(0.5, 0.5),
    float2(-0.5, 0.5)
};

static float3 colors[3] =
{
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0)
};

struct ConstantData
{
    float4 data;
};

ConstantBuffer<ConstantData> ConstantColor : register(b0, space0);

VSOutput VSmain(uint VertexIndex : SV_VertexID)
{
    VSOutput output = (VSOutput) 0;
    output.Pos = float4(positions[VertexIndex], 0.0, 1.0);
    return output;
}

float4 PSmain() : SV_TARGET
{
    return float4(ConstantColor.data.xyz, 1.f);
}