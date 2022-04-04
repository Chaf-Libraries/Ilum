#include "../Common.hlsli"

#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38
#define DBL_MAX 1.7976931348623158e+308
#define DBL_MIN 2.2250738585072014e-308
#define LOCAL_SIZE 64

StructuredBuffer<Instance> instances : register(t0);
RWStructuredBuffer<uint> visibilities : register(u1);
ConstantBuffer<Camera> camera : register(b2);
RWStructuredBuffer<DrawIndexedIndirectCommand> indirect_draws : register(u3);
ConstantBuffer<CullingInfo> culling_info : register(b4);
RWStructuredBuffer<CountInfo> count_info : register(u5);
RWStructuredBuffer<uint> draw_info : register(u6);

[[vk::push_constant]]
struct
{
    uint enable_frustum_culling;
} push_constants;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

bool CheckSphere(float3 pos, float radius)
{
    for (uint i = 0; i < 6; i++)
    {
        if (dot(camera.frustum[i], float4(pos, 1)) + radius < 0.0)
        {
            return false;
        }
    }
    return true;
}

bool CheckFrustum(Instance instance)
{
    float3 min_val = instance.bbox_min;
    float3 max_val = instance.bbox_max;

    float3 pos = (min_val + max_val) / 2.0;
    float radius = length(min_val - max_val) / 2.0;
    if (CheckSphere(pos, radius))
    {
        return true;
    }
    else
    {
        for (uint i = 0; i < 6; i++)
        {
            float4 plane = camera.frustum[i];
            float3 plane_normal = { plane.x, plane.y, plane.z };
            float plane_constant = plane.w;

            float3 axis_vert = { 0.0, 0.0, 0.0 };

            // x-axis
            axis_vert.x = plane.x < 0.0 ? min_val.x : max_val.x;

            // y-axis
            axis_vert.y = plane.y < 0.0 ? min_val.y : max_val.y;

            // z-axis
            axis_vert.z = plane.z < 0.0 ? min_val.z : max_val.z;

            if (dot(axis_vert, plane_normal) + plane_constant > 0.0)
            {
                return false;
            }
        }
    }

    return true;
}

[numthreads(LOCAL_SIZE, 1, 1)]
void main(CSParam param)
{
    uint idx = param.DispatchThreadID.x;
    uint temp = 0;

    if (idx == 0)
    {
        InterlockedExchange(count_info[0].instance_visible_count, 0, temp);
        InterlockedExchange(count_info[0].instance_invisible_count, 0, temp);
        InterlockedExchange(count_info[0].instance_total_count, 0, temp);
        InterlockedExchange(count_info[0].actual_draw, 0, temp);
        InterlockedExchange(count_info[0].total_draw, 0, temp);
    }

    if (idx >= culling_info.instance_count)
    {
        return;
    }

    Instance instance = instances[idx];

    float4x4 trans = transpose(instance.transform);

    float3 new_min_, new_max_, xa, xb, ya, yb, za, zb;

    xa = trans[0].xyz * instance.bbox_min[0];
    xb = trans[0].xyz * instance.bbox_max[0];

    ya = trans[1].xyz * instance.bbox_min[1];
    yb = trans[1].xyz * instance.bbox_max[1];

    za = trans[2].xyz * instance.bbox_min[2];
    zb = trans[2].xyz * instance.bbox_max[2];

    new_min_ = trans[3].xyz;
    new_min_ += min(xa, xb);
    new_min_ += min(ya, yb);
    new_min_ += min(za, zb);

    new_max_ = trans[3].xyz;
    new_max_ += max(xa, xb);
    new_max_ += max(ya, yb);
    new_max_ += max(za, zb);

    instance.bbox_min = new_min_;
    instance.bbox_max = new_max_;

    bool visible = true;

    if (push_constants.enable_frustum_culling == 1)
    {
        visible = CheckFrustum(instance);
    }

    if (visible)
    {
        uint dci;
        InterlockedAdd(count_info[0].instance_visible_count, 1, dci);
        indirect_draws[dci].indexCount = instance.index_count;
        indirect_draws[dci].instanceCount = 1;
        indirect_draws[dci].firstIndex = instance.index_offset;
        indirect_draws[dci].vertexOffset = int(instance.vertex_offset);
        indirect_draws[dci].firstInstance = idx;
        draw_info[dci] = idx;

        visibilities[idx] = 1;
        InterlockedAdd(count_info[0].actual_draw, 1, temp);
    }
    else
    {
        uint dci;
        InterlockedAdd(count_info[0].instance_invisible_count, 1, dci);
        indirect_draws[culling_info.instance_count - 1 - dci].indexCount = instance.index_count;
        indirect_draws[culling_info.instance_count - 1 - dci].instanceCount = 1;
        indirect_draws[culling_info.instance_count - 1 - dci].firstIndex = instance.index_offset;
        indirect_draws[culling_info.instance_count - 1 - dci].vertexOffset = int(instance.vertex_offset);
        indirect_draws[culling_info.instance_count - 1 - dci].firstInstance = idx;

        draw_info[culling_info.instance_count - 1 - dci] = idx;

        visibilities[idx] = 0;
    }
    
    InterlockedAdd(count_info[0].instance_total_count, 1, temp);
    InterlockedAdd(count_info[0].total_draw, 1, temp);
}