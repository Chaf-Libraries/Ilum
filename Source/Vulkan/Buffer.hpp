#pragma once

#include "Vulkan.hpp"

namespace Ilum::Vulkan
{
class Buffer
{
  public:
	Buffer(uint64_t                        size,
	       VkBufferUsageFlags              buffer_usage,
	       VmaMemoryUsage                  memory_usage,
	       VmaAllocationCreateFlags        flags          = VMA_ALLOCATION_CREATE_MAPPED_BIT,
	       const std::vector<QueueFamily> &queue_families = {});
	~Buffer();

	Buffer(const Buffer &) = delete;
	Buffer &operator=(const Buffer &) = delete;
	Buffer(Buffer &&)                 = delete;
	Buffer &operator=(Buffer &&) = delete;

	void     Flush() const;
	uint8_t *Map();
	void     Unmap();

	operator const VkBuffer &() const;

	const VkBuffer &GetHandle() const;
	uint64_t        GetSize() const;
	uint64_t        GetDeviceAddress();

	void Update(const uint8_t *data, size_t size, size_t offset = 0);
	void Update(const void *data, size_t size, size_t offset = 0);

	template <typename T>
	inline void Update(const T &data, size_t offset = 0)
	{
		Update(reinterpret_cast<const uint8_t *>(&data), sizeof(T), offset);
	}

	void SetName(const std::string &name);

  private:
	VkBuffer       m_handle      = VK_NULL_HANDLE;
	VmaAllocation  m_allocation  = VK_NULL_HANDLE;
	VkDeviceMemory m_memory      = VK_NULL_HANDLE;
	uint64_t       m_size        = 0;
	uint8_t *      m_mapped_data = nullptr;
	bool           m_persistent  = false;
	bool           m_mapped      = false;
};

// TODO: Do I need a buffer pool?
}        // namespace Ilum::Vulkan