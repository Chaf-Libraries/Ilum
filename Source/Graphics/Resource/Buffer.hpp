#pragma once

#include "../Vulkan.hpp"

namespace Ilum::Graphics
{
class Device;

class Buffer
{
  public:
	Buffer(const Device &device);
	Buffer(const Device &device, VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage);
	~Buffer();

	Buffer(const Buffer &) = delete;
	Buffer &operator=(const Buffer &) = delete;
	Buffer(Buffer &&other) noexcept;
	Buffer &operator=(Buffer &&other) noexcept;

	operator const VkBuffer &() const;

	const VkBuffer &GetHandle() const;
	VkDeviceSize    GetSize() const;

	bool     IsMapped() const;
	uint8_t *Map();
	void     Unmap();

	void Flush();
	void Flush(VkDeviceSize size, VkDeviceSize offset);

	void Copy(const uint8_t *data, VkDeviceSize size, VkDeviceSize offset);
	void CopyFlush(const uint8_t *data, VkDeviceSize size, VkDeviceSize offset);

  private:
	void Destroy();

  private:
	const Device &m_device;
	VkBuffer      m_handle     = VK_NULL_HANDLE;
	VmaAllocation m_allocation = VK_NULL_HANDLE;
	VkDeviceSize  m_size       = 0;
	uint8_t *     m_mapping    = nullptr;
};

using BufferReference = std::reference_wrapper<const Buffer>;

struct BufferInfo
{
	BufferReference handle;
	uint32_t        offset = 0;
};
}        // namespace Ilum::Graphics