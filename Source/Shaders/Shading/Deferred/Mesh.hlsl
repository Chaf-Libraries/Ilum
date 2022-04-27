#include "../../Common.hlsli"
#include "../../Math.hlsli"

#ifndef RUNTIME
#define TASK
#define MESH
#define FRAGMENT
#endif

StructuredBuffer<Instance> instances : register(t0);
StructuredBuffer<Meshlet> meshlets : register(t1);
ConstantBuffer<Camera> camera : register(b2);
StructuredBuffer<Vertex> vertices : register(t3);
ConstantBuffer<CullingInfo> culling_info : register(b5);
StructuredBuffer<uint> meshlet_vertices : register(t6);
StructuredBuffer<uint> meshlet_indices : register(t7);
RWStructuredBuffer<CountInfo> count_info : register(u8);
StructuredBuffer<MaterialData> materials : register(t9);
Texture2D textureArray[] : register(t10);
SamplerState texSampler : register(s11);

struct CSParam
{
    uint DispatchThreadID : SV_DispatchThreadID;
    uint GroupThreadID : SV_GroupThreadID;
    uint GroupID : SV_GroupID;
};

struct VertexOut
{
    float4 PositionHS : SV_Position;
    float4 ScreenPos : POSITIONT0;
    float4 PrevScreenPos : POSITIONT1;
    float3 Normal : NORMAL0;
    float2 TexCoord : COLOR0;
    uint EntityID : COLOR1;
    uint InstanceID : COLOR2;
};

struct MSOutput
{
    float4 GBuffer0 : SV_Target0; // GBuffer0: RGB - Albedo, A - metallic
    float4 GBuffer1 : SV_Target1; // GBuffer1: RGA - normal, A - linear depth
    float4 GBuffer2 : SV_Target2; // GBuffer2: RGB - emissive, A - roughness
    float4 GBuffer3 : SV_Target3; // GBuffer3: R - entity id, G - instance id, BA - motion vector
};

struct Payload
{
    uint meshletIndices[32];
};

#ifdef TASK
groupshared Payload shared_payload;
[numthreads(32, 1, 1)]
void ASmain(CSParam param)
{
    uint temp;
    if (param.DispatchThreadID.x == 0)
    {
        InterlockedExchange(count_info[0].meshlet_visible_count, 0, temp);
    }
    
    bool visible = false;
    
    if (param.DispatchThreadID.x < culling_info.meshlet_count)
    {
        Meshlet meshlet = meshlets[param.DispatchThreadID.x];
        Instance instance = instances[meshlet.instance_id];
        Camera cam = camera;
        visible = meshlet.IsVisible(cam, instance.transform);
    }
    
    if (visible)
    {
        uint index = WavePrefixCountBits(visible);
        shared_payload.meshletIndices[index] = param.DispatchThreadID.x;
        InterlockedAdd(count_info[0].meshlet_visible_count, 1, temp);
    }

    uint visible_count = WaveActiveCountBits(visible);
    
    DispatchMesh(visible_count, 1, 1, shared_payload);
}
#endif

#ifdef MESH
[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void MSmain(CSParam param, in payload Payload pay_load, out vertices VertexOut verts[64], out indices uint3 tris[126])
{
    
    uint meshlet_index = pay_load.meshletIndices[param.GroupID.x];
    
    if (meshlet_index >= culling_info.meshlet_count)
    {
        return;
    }
    
    Meshlet meshlet = meshlets[meshlet_index];
    Instance instance = instances[meshlet.instance_id];
    
    float4x4 transform = instance.transform;
    
    SetMeshOutputCounts(meshlet.vertex_count, meshlet.index_count / 3);
    
    for (uint i = param.GroupThreadID.x; i < meshlet.vertex_count; i += 32)
    {
        uint vertex_index = meshlet.vertex_offset + meshlet_vertices[meshlet.meshlet_vertex_offset + i];
        Vertex vertex = vertices[vertex_index];

        verts[i].Normal = normalize(mul((float3x3) transform, vertex.normal.xyz));
        verts[i].TexCoord = vertex.uv.rg;
        verts[i].EntityID = instance.entity_id;
        verts[i].InstanceID = meshlet.instance_id;
        
        float4 clipPos = mul(camera.view_projection, mul(transform, float4(vertex.position.xyz, 1.0)));
        float4 prevClipPos = mul(camera.last_view_projection, mul(instance.last_transform, float4(vertex.position.xyz, 1.0)));
        
        verts[i].PositionHS = clipPos;
        verts[i].ScreenPos = clipPos;
        verts[i].PrevScreenPos = prevClipPos;

    }

    for (i = param.GroupThreadID.x; i < meshlet.index_count / 3; i += 32)
    {
        for (int j = i * 3; j < i * 3 + 3; j++)
        {
            uint a = (meshlet.meshlet_index_offset + j) / 4;
            uint b = (meshlet.meshlet_index_offset + j) % 4;
            uint idx = (meshlet_indices[a] & 0x000000ffU << (8 * b)) >> (8 * b);
            tris[i][j % 3] = idx;
        }
    }
}
#endif

#ifdef FRAGMENT

float2 ComputeMotionVector(float4 prev_pos, float4 current_pos)
{
    // Clip space -> NDC
    float2 current = current_pos.xy / current_pos.w;
    float2 prev = prev_pos.xy / prev_pos.w;

    current = current * 0.5 + 0.5;
    prev = prev * 0.5 + 0.5;

    current.y = 1 - current.y;
    prev.y = 1 - prev.y;
    
    return current - prev;
}

MSOutput PSmain(VertexOut input)
{
    MSOutput output;
    
    float3 N = normalize(input.Normal);
    float3 T, B;
    CreateCoordinateSystem(N, T, B);
    float3x3 TBN = float3x3(T, B, N);
    
    // GBuffer0: RGB - Albedo
    if (materials[input.InstanceID].textures[TEXTURE_BASE_COLOR] < MAX_TEXTURE_ARRAY_SIZE)
    {
        output.GBuffer0.rgb = textureArray[NonUniformResourceIndex(materials[input.InstanceID].textures[TEXTURE_BASE_COLOR])].Sample(texSampler, input.TexCoord).rgb
                                        * materials[input.InstanceID].base_color.rgb;
    }
    else
    {
        output.GBuffer0.rgb = materials[input.InstanceID].base_color.rgb;
    }
    
    // GBuffer0: A - Metallic
    if (materials[input.InstanceID].textures[TEXTURE_METALLIC] < MAX_TEXTURE_ARRAY_SIZE)
    {
        output.GBuffer0.a = textureArray[NonUniformResourceIndex(materials[input.InstanceID].textures[TEXTURE_METALLIC])].Sample(texSampler, input.TexCoord).r
                                    * materials[input.InstanceID].metallic;
    }
    else
    {
        output.GBuffer0.a = materials[input.InstanceID].metallic;
    }
    
    // GBuffer1: RGB - Normal
    if (materials[input.InstanceID].textures[TEXTURE_NORMAL] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float3 normal = textureArray[NonUniformResourceIndex(materials[input.InstanceID].textures[TEXTURE_NORMAL])].Sample(texSampler, input.TexCoord).rgb;
        normal = normalize(mul(normalize(normal * 2.0 - float3(1.0, 1.0, 1.0)), TBN));
        output.GBuffer1.rgb = normal;

    }
    else
    {
        output.GBuffer1.rgb = N;
    }
        
    // GBuffer1: A - Linear Depth
    output.GBuffer1.a = input.ScreenPos.z;
    
    // GBuffer2: RGB - Emissive
    if (materials[input.InstanceID].textures[TEXTURE_EMISSIVE] < MAX_TEXTURE_ARRAY_SIZE)
    {
        output.GBuffer2.rgb = textureArray[NonUniformResourceIndex(materials[input.InstanceID].textures[TEXTURE_EMISSIVE])].Sample(texSampler, input.TexCoord).rgb
                                    * materials[input.InstanceID].emissive_color * materials[input.InstanceID].emissive_intensity;
    }
    else
    {
        output.GBuffer2.rgb = materials[input.InstanceID].emissive_color * materials[input.InstanceID].emissive_intensity;
    }
    
    // GBuffer2: A - Roughness
    if (materials[input.InstanceID].textures[TEXTURE_ROUGHNESS] < MAX_TEXTURE_ARRAY_SIZE)
    {
        output.GBuffer2.a = textureArray[NonUniformResourceIndex(materials[input.InstanceID].textures[TEXTURE_ROUGHNESS])].Sample(texSampler, input.TexCoord).g
                                    * materials[input.InstanceID].roughness;
    }
    else
    {
        output.GBuffer2.a = materials[input.InstanceID].roughness;
    }
    
    // GBuffer3: R - Entity ID
    output.GBuffer3.r = input.EntityID;
    
    // GBuffer3: G - Instance ID
    output.GBuffer3.g = input.InstanceID;
    
    // GBuffer3: BA - Motion Vector
    output.GBuffer3.ba = ComputeMotionVector(input.PrevScreenPos, input.ScreenPos);
   
    return output;
}
#endif