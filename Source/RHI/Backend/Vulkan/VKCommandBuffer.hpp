#pragma once

#include "Vulkan.hpp"

#include "../../RHICommand.hpp"

namespace Ilum::RHI::Vulkan
{
enum class CommandState
{
	Initial,
	Recording,
	Executable,
	Pending,
	Invalid
};

class VKCommandBuffer : public RHICommand
{
  public:
	VKCommandBuffer(const CmdUsage &usage, const std::thread::id &thread_id);

	virtual ~VKCommandBuffer() override;

	operator const VkCommandBuffer &() const;

	const VkCommandBuffer &GetHandle() const;

	const CommandState &GetState() const;

	virtual void Begin() override;

	virtual void End() override;

	virtual void Reset() override;

  private:
	VkCommandBuffer m_handle = VK_NULL_HANDLE;

	CommandState m_state = CommandState::Invalid;

	size_t m_pool_index = 0;
};
}        // namespace Ilum::RHI::Vulkan