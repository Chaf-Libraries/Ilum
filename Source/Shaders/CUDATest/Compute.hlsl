RWTexture2D<float4> Result;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
    uint3 GroupThreadID : SV_GroupThreadID;
    uint3 GroupID : SV_GroupID;
};

[numthreads(8, 8, 1)]
void MainCS(CSParam param)
{
    if (param.DispatchThreadID.x > 100 || param.DispatchThreadID.y > 100)
    {
        return;
    }
    
    float4 result = 0.f;
    for (uint i = 0; i < 0x00100000 >> 4; i++)
    {
        result.x = sin(result.x + param.DispatchThreadID.x * param.DispatchThreadID.x * (i * 4.0));
        result.y = cos(result.y + param.DispatchThreadID.y * param.DispatchThreadID.y * (i * 4.0 + 1.0));
    }
    
    result.w = 1.f;
    
    Result[param.DispatchThreadID.xy] = result;
}

