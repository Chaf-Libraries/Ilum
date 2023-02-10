#define MAX_BONE_INFLUENCE 8

struct VSInput
{
    float3 Position : POSITIONT0;
    int4 BoneID0 : BLENDINDICES0;
    int4 BoneID1 : BLENDINDICES1;
    float4 BoneWeight0 : BLENDWEIGHT0;
    float4 BoneWeight1 : BLENDWEIGHT1;
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
    
    uint bone_count = 0;
    uint bone_stride = 0;
    BoneMatrices.GetDimensions(bone_count, bone_stride);
    
    float4 total_position = 0.f;
    for (uint i = 0; i < MAX_BONE_INFLUENCE; i++)
    {
        int bone = -1;
        float weight = 0.f;
        
        if (i < 4)
        {
            bone = input.BoneID0[i];
            weight = input.BoneWeight0[i];
        }
        else
        {
            bone = input.BoneID1[i - 4];
            weight = input.BoneWeight1[i - 4];
        }
        
        if (bone == -1)
        {
            continue;
        }
        
        if (bone >= bone_count)
        {
            total_position = float4(input.Position, 1.0f);
            break;
        }
        
        float4 local_position = mul(BoneMatrices[bone], float4(input.Position, 1.0f));
        total_position += local_position * weight;
    }
        
    output.Position = mul(UniformBuffer.transform, total_position);
    output.Color = GenerateColor(input.BoneID0[0]);
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
