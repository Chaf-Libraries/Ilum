#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class CommandBuffer;

class Buffer
{
  public:
	Buffer() = default;

	Buffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage);

	~Buffer();

	Buffer(const Buffer &) = delete;

	Buffer &operator=(const Buffer &) = delete;

	Buffer(Buffer &&other);

	Buffer &operator=(Buffer &&other);

	const VkBuffer &getBuffer() const;

	operator const VkBuffer &() const;

	VkDeviceSize getSize() const;

	uint64_t getDeviceAddress() const;

	bool isMapped() const;

	uint8_t *map();

	void unmap();

	void flush();

	void flush(VkDeviceSize size, VkDeviceSize offset);

	void copy(const uint8_t *data, VkDeviceSize size, VkDeviceSize offset);

	void copyFlush(const uint8_t *data, VkDeviceSize size, VkDeviceSize offset);

  private:
	void destroy();

  private:
	VkBuffer      m_handle     = VK_NULL_HANDLE;
	VmaAllocation m_allocation = VK_NULL_HANDLE;
	VkDeviceSize  m_size       = 0;
	uint8_t *     m_mapping    = nullptr;
	uint64_t      m_device_address = 0;
};

using BufferReference = std::reference_wrapper<const Buffer>;
}        // namespace Ilum