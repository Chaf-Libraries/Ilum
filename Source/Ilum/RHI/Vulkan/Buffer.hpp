#pragma once

#include "RHI/RHIBuffer.hpp"

#include <vk_mem_alloc.h>
#include <volk.h>

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

	static BufferState Create(RHIBufferState state);
};

class Buffer : public RHIBuffer
{
  public:
	Buffer(RHIDevice *device, const BufferDesc &desc);
	virtual ~Buffer() override;

	virtual void *Map() override;
	virtual void  Unmap() override;

	VkBuffer GetHandle() const;

	uint64_t GetDeviceAddress() const;

  private:
	VkBuffer      m_handle     = VK_NULL_HANDLE;
	VmaAllocation m_allocation = VK_NULL_HANDLE;

	uint64_t m_device_address = 0;

	void *m_mapped = nullptr;
};
}        // namespace Ilum::Vulkan