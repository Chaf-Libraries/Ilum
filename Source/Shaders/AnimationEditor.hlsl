//#define SHADING
//#define DRAW_NORMAL
//#define DRAW_UV
//#define DRAW_TEXTURE
//#define WIREFRAME

struct VSInput
{
    uint BoneID : BLENDINDICES0;
};

struct UniformBlock
{
    float4x4 transform;
};

ConstantBuffer<UniformBlock> UniformBuffer;
StructuredBuffer<float4x4> BoneMatrices;

struct VSOutput
{
    float4 Position : SV_Position;
    float4 Color : COLOR0;
    float PointSize : PSIZE;
};

uint Hash(uint a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

float4 GenerateColor(uint a)
{
    uint hash = Hash(a);
    return float4(float3(float(hash & 255), float((hash >> 8) & 255), float((hash >> 16) & 255)) / 255.0, 1.0);
}

VSOutput VSmain(VSInput input)
{
    VSOutput output = (VSOutput) 0;
    output.Position = mul(UniformBuffer.transform, mul(BoneMatrices[input.BoneID], float4(0.f, 0.f, 0.f, 1.0f)));
    output.Color = GenerateColor(input.BoneID);
    return output;
}

struct PSInput
{
    float4 Color : COLOR0;
};

float4 PSmain(PSInput input) : SV_TARGET
{
    return input.Color;
}
