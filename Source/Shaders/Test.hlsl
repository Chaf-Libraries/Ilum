#define LOCAL_SIZE 32
#define LUT_SIZE 1024
#define SAMPLE_COUNT 4096

struct AddressData
{
    uint64_t addr[3];
};

RWTexture2D<float4> Result : register(u0);
//ConstantBuffer<AddressData> Address : register(b1);

[[vk::push_constant]]
struct
{
    uint color_index;
    uint64_t buffer_index;
} push_constant;

float3 FetchFloat3(uint64_t buffer_address, uint color_index)
{
    float3 foo;
    foo.x = asfloat(vk::RawBufferLoad(buffer_address + 12 * color_index));
    foo.y = asfloat(vk::RawBufferLoad(buffer_address + 12 * color_index + 4));
    foo.z = asfloat(vk::RawBufferLoad(buffer_address + 12 * color_index + 8));
    return foo;
}

struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

[numthreads(LOCAL_SIZE, LOCAL_SIZE, 1)]
void main(CSParam param)
{    
    Result[param.DispatchThreadID.xy] = float4(FetchFloat3(push_constant.buffer_index, push_constant.color_index), 1.0);

}