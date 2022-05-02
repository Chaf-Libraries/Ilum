#pragma once

#include <volk.h>

#include <vk_mem_alloc.h>

#include <string>

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

class Buffer
{
  public:
	Buffer(RHIDevice *device, const BufferDesc &desc);
	~Buffer();

	Buffer(const Buffer &) = delete;
	Buffer &operator=(const Buffer &) = delete;
	Buffer(Buffer &&other)            = delete;
	Buffer &operator=(Buffer &&other) = delete;

	VkDeviceSize       GetSize() const;
	VkBufferUsageFlags GetUsage() const;
	uint64_t           GetDeviceAddress() const;

	void *Map();
	void  Unmap();
	void  Flush(VkDeviceSize size, VkDeviceSize offset);

	operator VkBuffer() const;

	void SetName(const std::string &name);

  private:
	RHIDevice    *p_device;
	BufferDesc    m_desc;
	VkBuffer      m_handle;
	VmaAllocation m_allocation;
	uint64_t      m_device_address;
	void	     *m_mapped_data = nullptr;
};
using BufferReference = std::reference_wrapper<Buffer>;
}        // namespace Ilum