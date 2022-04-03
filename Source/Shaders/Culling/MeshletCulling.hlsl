#include "../Common.hlsli"

#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38
#define DBL_MAX 1.7976931348623158e+308
#define DBL_MIN 2.2250738585072014e-308
#define LOCAL_SIZE 64

RWStructuredBuffer<DrawIndexedIndirectCommand> indirect_draws : register(u0);
StructuredBuffer<Instance> instances : register(t1);
StructuredBuffer<Meshlet> meshlets : register(t2);
RWStructuredBuffer<uint> draw_info : register(u3);
ConstantBuffer<Camera> camera : register(b4);
Texture2D hizbuffer : register(t5);
SamplerState hizSampler : register(s5);
RWStructuredBuffer<CountInfo> count_info : register(u6);
ConstantBuffer<CullingInfo> culling_info : register(b7);
RWStructuredBuffer<uint> visibilities : register(u8);

[[vk::push_constant]]
struct
{
    uint enable_frustum_culling;
    uint enable_backface_culling;
    uint enable_occlusion_culling;
} push_constants;

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

bool CheckFrustum(Meshlet meshlet)
{
    for (uint i = 0; i < 6; i++)
    {
        if (dot(camera.frustum[i], float4(meshlet.center, 1)) + length(meshlet.radius) < 0.0)
        {
            return false;
        }
    }
    return true;
}

bool ProjectSphere(float3 C, float r, float znear, float P00, float P11, out float4 aabb)
{
    if (-C.z < r + znear)
        return false;

    float2 cx = C.xz;
    float2 vx = float2(sqrt(dot(cx, cx) - r * r), r);
    float2 minx = mul(float2x2(vx.x, vx.y, -vx.y, vx.x), cx);
    float2 maxx = mul(float2x2(vx.x, -vx.y, vx.y, vx.x), cx);

    float2 cy = C.yz;
    float2 vy = float2(sqrt(dot(cy, cy) - r * r), r);
    float2 miny = mul(float2x2(vy.x, vy.y, -vy.y, vy.x), cy);
    float2 maxy = mul(float2x2(vy.x, -vy.y, vy.y, vy.x), cy);

    aabb = float4(-minx.x / minx.y * P00, -miny.x / miny.y * P11, -maxx.x / maxx.y * P00, -maxy.x / maxy.y * P11);
    aabb = aabb.xwzy * float4(0.5f, -0.5f, 0.5f, -0.5f) + float4(0.5f, 0.5f, 0.5f, 0.5f);

    return true;
}

bool CheckOcclusion(Meshlet meshlet)
{
    float3 dir = normalize(camera.position - meshlet.center);
    float3 sceen_space_center_last = mul(camera.last_view_projection, float4(meshlet.center, 1.0)).xyz;

    float3 C = (mul(culling_info.view, float4(meshlet.center, 1.0))).xyz;

    float4 aabb;
    if (!ProjectSphere(C, meshlet.radius, culling_info.znear, culling_info.P00, culling_info.P11, aabb))
    {
        return true;
    }

    float width = (aabb.z - aabb.x) * culling_info.zbuffer_width;
    float height = (aabb.w - aabb.y) * culling_info.zbuffer_height;

    float mip_level = floor(log2(max(width, height))) - 1;

    float2 uv = (aabb.xy + aabb.zw) * 0.5;
    float2 uv0 = aabb.xy;
    float2 uv1 = aabb.zw;
    float2 uv2 = aabb.xw;
    float2 uv3 = aabb.zy;
    
    float depth = hizbuffer.SampleLevel(hizSampler, uv, mip_level).r;
    depth = max(depth, hizbuffer.SampleLevel(hizSampler, uv0, mip_level).r);
    depth = max(depth, hizbuffer.SampleLevel(hizSampler, uv1, mip_level).r);
    depth = max(depth, hizbuffer.SampleLevel(hizSampler, uv2, mip_level).r);
    depth = max(depth, hizbuffer.SampleLevel(hizSampler, uv3, mip_level).r);
    
    float depthSphere = abs(sceen_space_center_last.z) - 2 * meshlet.radius;

    return depth >= depthSphere;
}

[numthreads(LOCAL_SIZE, 1, 1)]
void main(CSParam param)
{
    uint idx = param.DispatchThreadID.x;
    uint temp = 0;
    if (idx == 0)
    {        
        InterlockedExchange(count_info[0].meshlet_visible_count, 0, temp);
        InterlockedExchange(count_info[0].meshlet_invisible_count, 0, temp);
        InterlockedExchange(count_info[0].meshlet_total_count, 0, temp);
        InterlockedExchange(count_info[0].actual_draw, 0, temp);
        InterlockedExchange(count_info[0].total_draw, 0, temp);
    }

    if (idx >= culling_info.meshlet_count)
    {
        return;
    }

    Meshlet meshlet = meshlets[idx];

    if (visibilities[meshlet.instance_id] != 1)
    {
        uint dci;
        InterlockedAdd(count_info[0].meshlet_invisible_count, 1, dci);
        indirect_draws[culling_info.meshlet_count - 1 - dci].indexCount = meshlet.index_count;
        indirect_draws[culling_info.meshlet_count - 1 - dci].instanceCount = 1;
        indirect_draws[culling_info.meshlet_count - 1 - dci].firstIndex = meshlet.index_offset;
        indirect_draws[culling_info.meshlet_count - 1 - dci].vertexOffset = int(meshlet.vertex_offset);
        indirect_draws[culling_info.meshlet_count - 1 - dci].firstInstance = 0;
        draw_info[culling_info.meshlet_count - 1 - dci] = meshlet.instance_id;
        InterlockedAdd(count_info[0].meshlet_total_count, 1, temp);
        return;
    }
    
    float4x4 trans = instances[meshlet.instance_id].transform;
    trans = transpose(trans);
    
    meshlet.center = mul(instances[meshlet.instance_id].transform, float4(meshlet.center, 1.0)).xyz;
    float3 edge = float3(1.0, 1.0, 1.0) * sqrt(meshlet.radius * meshlet.radius / 3.0);
    meshlet.radius = length(float3(
        abs(trans[0][0]) * edge.x + abs(trans[0][1]) * edge.y + abs(trans[0][2]) * edge.z,
        abs(trans[1][0]) * edge.x + abs(trans[1][1]) * edge.y + abs(trans[1][2]) * edge.z,
        abs(trans[2][0]) * edge.x + abs(trans[2][1]) * edge.y + abs(trans[2][2]) * edge.z
    ));

    bool visible = true;

    if (visible && push_constants.enable_frustum_culling == 1)
    {
        visible = visible && CheckFrustum(meshlet);
    }

    if (visible && push_constants.enable_backface_culling == 1)
    {
        float3 scale = float3(
            sqrt(trans[0][0] * trans[0][0] + trans[0][1] * trans[0][1] + trans[0][2] * trans[0][2]),
            sqrt(trans[1][0] * trans[1][0] + trans[1][1] * trans[1][1] + trans[1][2] * trans[1][2]),
            sqrt(trans[2][0] * trans[2][0] + trans[2][1] * trans[2][1] + trans[2][2] * trans[2][2])
        );

        float3x3 rotation = float3x3(
            trans[0].xyz / scale.x,
            trans[1].xyz / scale.y,
            trans[2].xyz / scale.z
        );

        meshlet.cone_apex = mul(instances[meshlet.instance_id].transform, float4(meshlet.cone_apex, 1.0)).xyz;
        meshlet.cone_axis = mul(rotation, meshlet.cone_axis);

        visible = visible && dot(normalize(camera.position - meshlet.cone_apex), meshlet.cone_axis) < meshlet.cone_cutoff;
    }

    if (visible && push_constants.enable_occlusion_culling == 1)
    {
        visible = visible && CheckOcclusion(meshlet);
    }

    if (visible)
    {
        uint dci;
        InterlockedAdd(count_info[0].meshlet_visible_count, 1, dci);
        indirect_draws[dci].indexCount = meshlet.index_count;
        indirect_draws[dci].instanceCount = 1;
        indirect_draws[dci].firstIndex = meshlet.index_offset;
        indirect_draws[dci].vertexOffset = int(meshlet.vertex_offset);
        indirect_draws[dci].firstInstance = meshlet.instance_id;
        draw_info[dci] = meshlet.instance_id;

        InterlockedAdd(count_info[0].actual_draw, 1, temp);
    }
    else
    {
        uint dci;
        InterlockedAdd(count_info[0].meshlet_invisible_count, 1, dci);
        indirect_draws[culling_info.meshlet_count - 1 - dci].indexCount = meshlet.index_count;
        indirect_draws[culling_info.meshlet_count - 1 - dci].instanceCount = 1;
        indirect_draws[culling_info.meshlet_count - 1 - dci].firstIndex = meshlet.index_offset;
        indirect_draws[culling_info.meshlet_count - 1 - dci].vertexOffset = int(meshlet.vertex_offset);
        indirect_draws[culling_info.meshlet_count - 1 - dci].firstInstance = meshlet.instance_id;
        draw_info[culling_info.meshlet_count - 1 - dci] = meshlet.instance_id;
    }
    
    InterlockedAdd(count_info[0].meshlet_total_count, 1, temp);
    InterlockedAdd(count_info[0].total_draw, 1, temp);
}