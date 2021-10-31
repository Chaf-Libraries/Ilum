#pragma once

#include <Utils/PCH.hpp>

#include "Graphics/Synchronization/QueueSystem.hpp"

namespace Ilum
{
class LogicalDevice;

class CommandPool
{
  public:
	CommandPool(QueueUsage usage = QueueUsage::Graphics, const std::thread::id &thread_id = std::this_thread::get_id());

	~CommandPool();

	void reset();

	operator const VkCommandPool &() const;

	const VkCommandPool &getCommandPool() const;

	const std::thread::id &getThreadID() const;

	QueueUsage getUsage() const;

  private:
	VkCommandPool   m_handle = VK_NULL_HANDLE;
	std::thread::id m_thread_id;
	QueueUsage      m_queue_usage = QueueUsage::Graphics;
};
}        // namespace Ilum