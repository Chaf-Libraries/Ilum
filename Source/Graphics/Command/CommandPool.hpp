#pragma once

#include <Utils/PCH.hpp>

namespace Ilum
{
class LogicalDevice;

class CommandPool
{
  public:
	CommandPool(VkQueueFlagBits queue_type = VK_QUEUE_GRAPHICS_BIT, const std::thread::id &thread_id = std::this_thread::get_id());

	~CommandPool();

	void reset();

	operator const VkCommandPool &() const;

	const VkCommandPool &getCommandPool() const;

	const std::thread::id &getThreadID() const;

	const VkQueue getQueue(uint32_t index) const;

  private:
	const LogicalDevice &m_logical_device;
	VkCommandPool        m_handle = VK_NULL_HANDLE;
	std::thread::id      m_thread_id;
	VkQueueFlagBits      m_queue_type = VK_QUEUE_GRAPHICS_BIT;
};
}        // namespace Ilum