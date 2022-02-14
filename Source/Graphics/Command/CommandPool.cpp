#include "CommandPool.hpp"
#include "CommandBuffer.hpp"

#include <Core/Hash.hpp>

#include <Graphics/Device/Device.hpp>
#include <Graphics/RenderContext.hpp>

namespace Ilum::Graphics
{
CommandPool::CommandPool(const Device &device, QueueFamily queue, ResetMode reset_mode, const std::thread::id &thread_id) :
    m_device(device),
    m_queue(queue),
    m_thread_id(thread_id),
    m_reset_mode(reset_mode)
{
	m_hash = 0;
	Core::HashCombine(m_hash, static_cast<size_t>(m_queue));
	Core::HashCombine(m_hash, static_cast<size_t>(reset_mode));
	Core::HashCombine(m_hash, thread_id);

	VkCommandPoolCreateInfo create_info = {};
	create_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	create_info.flags                   = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
	create_info.queueFamilyIndex        = RenderContext::GetDevice().GetQueueFamily(m_queue);

	switch (reset_mode)
	{
		case ResetMode::ResetIndividually:
		case ResetMode::AlwaysAllocate:
			create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			break;
		case ResetMode::ResetPool:
		default:
			create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
			break;
	}

	vkCreateCommandPool(RenderContext::GetDevice(), &create_info, nullptr, &m_handle);
}

CommandPool::~CommandPool()
{
	m_primary_cmd_buffers.clear();
	m_secondary_cmd_buffers.clear();
	vkDestroyCommandPool(RenderContext::GetDevice(), m_handle, nullptr);
}

CommandPool::operator const VkCommandPool &() const
{
	return m_handle;
}

const VkCommandPool &CommandPool::GetHandle() const
{
	return m_handle;
}

const std::thread::id &CommandPool::GetThreadID() const
{
	return m_thread_id;
}

const QueueFamily &CommandPool::GetQueueFamily() const
{
	return m_queue;
}

size_t CommandPool::GetHash() const
{
	return m_hash;
}

CommandPool::ResetMode CommandPool::GetResetMode() const
{
	return m_reset_mode;
}

void CommandPool::Reset()
{
	switch (m_reset_mode)
	{
		case ResetMode::ResetIndividually:
			for (auto &cmd_buffer : m_primary_cmd_buffers)
			{
				cmd_buffer->Reset();
			}
			for (auto &cmd_buffer : m_secondary_cmd_buffers)
			{
				cmd_buffer->Reset();
			}
			break;
		case ResetMode::ResetPool:
			vkResetCommandPool(RenderContext::GetDevice(), m_handle, 0);
			for (auto &cmd_buffer : m_primary_cmd_buffers)
			{
				cmd_buffer->Reset();
			}
			for (auto &cmd_buffer : m_secondary_cmd_buffers)
			{
				cmd_buffer->Reset();
			}
			break;
		case ResetMode::AlwaysAllocate:
			m_primary_cmd_buffers.clear();
			m_secondary_cmd_buffers.clear();
			break;
		default:
			throw std::runtime_error("Unknown reset mode for command pools");
	}

	m_active_primary_count = 0;
	m_active_secondary_count = 0;
}

CommandBuffer& CommandPool::RequestCommandBuffer(VkCommandBufferLevel level)
{
	if (level == VK_COMMAND_BUFFER_LEVEL_PRIMARY)
	{
		if (m_active_primary_count < m_primary_cmd_buffers.size())
		{
			return *m_primary_cmd_buffers.at(m_active_primary_count++);
		}

		m_primary_cmd_buffers.emplace_back(std::make_unique<CommandBuffer>(*this, level));

		m_active_primary_count++;

		return *m_primary_cmd_buffers.back();
	}
	else
	{
		if (m_active_secondary_count < m_secondary_cmd_buffers.size())
		{
			return *m_secondary_cmd_buffers.at(m_active_secondary_count++);
		}

		m_secondary_cmd_buffers.emplace_back(std::make_unique<CommandBuffer>(*this, level));

		m_active_secondary_count++;

		return *m_secondary_cmd_buffers.back();
	}
}
}        // namespace Ilum::Graphics