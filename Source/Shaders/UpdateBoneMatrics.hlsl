struct UpdateInfoData
{
    uint count;
    float time;
};

Texture2D<float4> SkinnedMatrics;
RWStructuredBuffer<float4x4> BoneMatrics;
ConstantBuffer<UpdateInfoData> UpdateInfo;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

[numthreads(8, 1, 1)]
void CSmain(CSParam param)
{
    uint bone_id = param.DispatchThreadID.x;
    
    uint bone_count = 0;
    uint frame_count = 0;
    uint level_count = 0;
    SkinnedMatrics.GetDimensions(0, bone_count, frame_count, level_count);
    bone_count /= 3;
        
    if (bone_id >= bone_count)
    {
        return;
    }
    
    uint frame0 = min(uint(UpdateInfo.time * 30.f), frame_count - 1);
    uint frame1 = min(uint(UpdateInfo.time * 30.f) + 1, frame_count - 1);
        
    float4x4 mat0 = float4x4(
        SkinnedMatrics.Load(int3(bone_id * 3 + 0, frame0, 0)),
        SkinnedMatrics.Load(int3(bone_id * 3 + 1, frame0, 0)),
        SkinnedMatrics.Load(int3(bone_id * 3 + 2, frame0, 0)),
        float4(0, 0, 0, 1));

    float4x4 mat1 = float4x4(
        SkinnedMatrics.Load(int3(bone_id * 3 + 0, frame1, 0)),
        SkinnedMatrics.Load(int3(bone_id * 3 + 1, frame1, 0)),
        SkinnedMatrics.Load(int3(bone_id * 3 + 2, frame1, 0)),
        float4(0, 0, 0, 1));
    
    BoneMatrics[bone_id] = lerp(mat0, mat1, frac(UpdateInfo.time * 30.f));
}