#pragma once

#include <Core/Engine/PCH.hpp>

namespace Ilum
{
class LogicalDevice;

class CommandPool
{
  public:
	enum class Usage
	{
		Graphics,
		Compute,
		Transfer
	};

  public:
	CommandPool(const LogicalDevice &logical_device, Usage usage = Usage::Graphics, const std::thread::id &thread_id = std::this_thread::get_id());

	~CommandPool();

	void reset();

	operator const VkCommandPool &() const;

	const LogicalDevice &getLogicalDevice() const;

	const VkCommandPool &getCommandPool() const;

	const std::thread::id &getThreadID() const;

	const VkQueue getQueue(uint32_t index) const;

  private:
	const LogicalDevice &m_logical_device;
	VkCommandPool        m_handle = VK_NULL_HANDLE;
	std::thread::id      m_thread_id;
	Usage                m_usage = Usage::Graphics;
};
}        // namespace Ilum