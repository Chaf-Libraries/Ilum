#ifndef __FETCH_HLSL__
#define __FETCH_HLSL__

#include "Common.hlsli"

Vertex FetchVertex(uint64_t addr, uint index)
{
    Vertex vertex;
    vertex.position.x = asfloat(vk::RawBufferLoad(addr + 64 * index + 0));
    vertex.position.y = asfloat(vk::RawBufferLoad(addr + 64 * index + 4));
    vertex.position.z = asfloat(vk::RawBufferLoad(addr + 64 * index + 8));
    vertex.position.w = asfloat(vk::RawBufferLoad(addr + 64 * index + 12));
    vertex.uv.x = asfloat(vk::RawBufferLoad(addr + 64 * index + 16));
    vertex.uv.y = asfloat(vk::RawBufferLoad(addr + 64 * index + 20));
    vertex.uv.z = asfloat(vk::RawBufferLoad(addr + 64 * index + 24));
    vertex.uv.w = asfloat(vk::RawBufferLoad(addr + 64 * index + 28));
    vertex.normal.x = asfloat(vk::RawBufferLoad(addr + 64 * index + 32));
    vertex.normal.y = asfloat(vk::RawBufferLoad(addr + 64 * index + 36));
    vertex.normal.z = asfloat(vk::RawBufferLoad(addr + 64 * index + 40));
    vertex.normal.w = asfloat(vk::RawBufferLoad(addr + 64 * index + 44));
    vertex.tangent.x = asfloat(vk::RawBufferLoad(addr + 64 * index + 48));
    vertex.tangent.y = asfloat(vk::RawBufferLoad(addr + 64 * index + 52));
    vertex.tangent.z = asfloat(vk::RawBufferLoad(addr + 64 * index + 56));
    vertex.tangent.w = asfloat(vk::RawBufferLoad(addr + 64 * index + 60));
    return vertex;
}

#endif