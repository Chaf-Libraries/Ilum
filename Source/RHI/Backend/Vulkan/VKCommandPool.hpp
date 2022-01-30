#pragma once

#include "Vulkan.hpp"

#include "../../RHICommand.hpp"

namespace Ilum::RHI::Vulkan
{
class VKCommandPool
{
  public:
	VKCommandPool(const CmdUsage &usage, const std::thread::id &thread_id);

	~VKCommandPool();

	VKCommandPool(const VKCommandPool &) = delete;

	VKCommandPool &operator=(VKCommandPool &) = delete;

	VKCommandPool(VKCommandPool &&) = delete;

	VKCommandPool &operator=(VKCommandPool &&) = delete;

	operator const VkCommandPool &() const;

	const VkCommandPool &GetHandle() const;

	void Reset();

	const std::thread::id &GetThreadID() const;

	const CmdUsage &GetUsage() const;

	size_t GetHash() const;

  private:
	VkCommandPool   m_handle = VK_NULL_HANDLE;
	std::thread::id m_thread_id;
	CmdUsage        m_usage;
	size_t          m_hash = 0;
};
}        // namespace Ilum::RHI::Vulkan