#pragma once

#include "RHI/RHIBuffer.hpp"

#include <volk.h>

#include <vk_mem_alloc.h>

namespace Ilum::Vulkan
{
struct BufferState
{
	VkAccessFlags        access_mask = VK_ACCESS_NONE;
	VkPipelineStageFlags stage       = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;

	inline bool operator==(const BufferState &other)
	{
		return access_mask == other.access_mask &&
		       stage == other.stage;
	}

	static BufferState Create(RHIResourceState state);
};

class Buffer : public RHIBuffer
{
  public:
	Buffer(RHIDevice *device, const BufferDesc &desc);

	virtual ~Buffer() override;

	virtual void CopyToDevice(const void *data, size_t size, size_t offset = 0) override;

	virtual void CopyToHost(void *data, size_t size, size_t offset) override;

	virtual void *Map() override;

	virtual void Unmap() override;

	virtual void Flush(size_t offset, size_t size) override;

	VkBuffer GetHandle() const;

	VkDeviceMemory GetMemory() const;

	uint64_t GetDeviceAddress() const;

  private:
	VkBuffer      m_handle     = VK_NULL_HANDLE;
	VmaAllocation m_allocation = VK_NULL_HANDLE;

	uint64_t m_device_address = 0;

	void *m_mapped = nullptr;
};
}        // namespace Ilum::Vulkan