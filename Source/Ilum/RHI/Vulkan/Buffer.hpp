#pragma once

#include "RHI/RHIBuffer.hpp"

#include <vk_mem_alloc.h>
#include <volk.h>

namespace Ilum::Vulkan
{
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