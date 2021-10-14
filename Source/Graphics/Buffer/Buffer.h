#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class CommandBuffer;

class Buffer
{
  public:
	Buffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage);

	~Buffer();

	const VkBuffer& getBuffer() const;

	operator const VkBuffer &() const;

	VkDeviceSize getSize() const;

	bool isMapped() const;

	uint8_t *map();

	void unmap();

	void flush();

	void flush(VkDeviceSize size, VkDeviceSize offset);

	void copy(const uint8_t *data, VkDeviceSize size, VkDeviceSize offset);

	void copyFlush(const uint8_t *data, VkDeviceSize size, VkDeviceSize offset);

  private:
	VkBuffer      m_handle     = VK_NULL_HANDLE;
	VmaAllocation m_allocation = VK_NULL_HANDLE;
	VkDeviceSize  m_size       = 0;
	uint8_t *     m_mapping    = nullptr;
};

using BufferReference = std::reference_wrapper<const Buffer>;
}        // namespace Ilum