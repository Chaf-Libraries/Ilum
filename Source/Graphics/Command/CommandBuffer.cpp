#include "CommandBuffer.hpp"
#include "CommandPool.hpp"

namespace Ilum::Graphics
{
CommandBuffer::CommandBuffer(const CommandPool &cmd_pool, VkCommandBufferLevel level) :
    m_pool(cmd_pool), m_level(level)
{
}

CommandBuffer::~CommandBuffer()
{
}

CommandBuffer::operator const VkCommandBuffer &() const
{
	return m_handle;
}

const VkCommandBuffer &CommandBuffer::GetHandle() const
{
	return m_handle;
}

const CommandBuffer::State &CommandBuffer::GetState() const
{
	return m_state;
}

VkCommandBufferLevel CommandBuffer::GetLevel() const
{
	return m_level;
}

void CommandBuffer::Reset()
{
}

void CommandBuffer::Begin(VkCommandBufferUsageFlagBits usage, VkCommandBufferInheritanceInfo *inheritanceInfo)
{
}

void CommandBuffer::End()
{
}

void CommandBuffer::SubmitIdle(uint32_t queue_index)
{
}
}        // namespace Ilum::Graphics