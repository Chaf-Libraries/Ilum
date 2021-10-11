#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class CommandBuffer;

class Buffer
{
  public:
	Buffer(VkDeviceSize size, VkBufferUsageFlags buffer_usage, VmaMemoryUsage memory_usage, const void *data = nullptr);

	virtual ~Buffer();

	operator const VkBuffer &() const;

	VkResult map(void **data) const;

	void unmap() const;

	const VkBuffer &getBuffer() const;

	const VkDeviceSize getSize() const;

	const VkDescriptorBufferInfo& getDescriptor() const;

  public:
	static void insertBufferMemoryBarrier(
	    const CommandBuffer &command_buffer,
	    const VkBuffer &     buffer,
	    VkAccessFlags        src_access_mask,
	    VkAccessFlags        dst_access_mask,
	    VkPipelineStageFlags src_stage_mask,
	    VkPipelineStageFlags dst_stage_mask,
	    VkDeviceSize         offset = 0,
	    VkDeviceSize         size   = VK_WHOLE_SIZE);

  private:
	void updateDescriptor();

  private:
	VkBuffer      m_handle     = VK_NULL_HANDLE;
	VmaAllocation m_allocation = VK_NULL_HANDLE;

	VkDeviceSize m_size = 0;

	VkDescriptorBufferInfo m_descriptor = {};
	// TODO: Dynamic buffer
};
}        // namespace Ilum