#pragma once

#include <cstdint>
#include <vk_mem_alloc.h>
#include <volk.h>

namespace Ilum
{
class RHIDevice;

struct BufferDesc
{
	BufferDesc() = default;
	BufferDesc(uint64_t elements, uint32_t elements_size, VkBufferUsageFlags buffer_usage, VmaMemoryUsage memory_usage) :
	    size(elements * elements_size), buffer_usage(buffer_usage), memory_usage(memory_usage)
	{}

	VkDeviceSize       size;
	VkBufferUsageFlags buffer_usage;
	VmaMemoryUsage     memory_usage;
};

struct Buffer
{
	Buffer(RHIDevice *device, const BufferDesc &desc, VkBuffer handle, VmaAllocation allocation);

};

}        // namespace Ilum