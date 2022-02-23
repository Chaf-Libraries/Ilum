#include "CommandPool.hpp"
#include "CommandBuffer.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Device/LogicalDevice.hpp"

#include "Utils/Hash.hpp"

namespace Ilum
{
CommandPool::CommandPool(QueueUsage queue, ResetMode reset_mode, const std::thread::id &thread_id) :
    m_queue(queue),
    m_thread_id(thread_id),
    m_reset_mode(reset_mode)
{
	m_hash = 0;
	hash_combine(m_hash, static_cast<size_t>(m_queue));
	hash_combine(m_hash, static_cast<size_t>(reset_mode));
	hash_combine(m_hash, thread_id);

	VkCommandPoolCreateInfo create_info = {};
	create_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	create_info.flags                   = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
	switch (m_queue)
	{
		case Ilum::QueueUsage::Graphics:
			create_info.queueFamilyIndex = GraphicsContext::instance()->getLogicalDevice().getGraphicsFamily();
			break;
		case Ilum::QueueUsage::Compute:
			create_info.queueFamilyIndex = GraphicsContext::instance()->getLogicalDevice().getComputeFamily();
			break;
		case Ilum::QueueUsage::Transfer:
			create_info.queueFamilyIndex = GraphicsContext::instance()->getLogicalDevice().getTransferFamily();
			break;
		case Ilum::QueueUsage::Present:
			create_info.queueFamilyIndex = GraphicsContext::instance()->getLogicalDevice().getPresentFamily();
		default:
			break;
	}

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

	vkCreateCommandPool(GraphicsContext::instance()->getLogicalDevice(), &create_info, nullptr, &m_handle);
}

CommandPool::~CommandPool()
{
	m_primary_cmd_buffers.clear();
	m_secondary_cmd_buffers.clear();
	vkDestroyCommandPool(GraphicsContext::instance()->getLogicalDevice(), m_handle, nullptr);
}

CommandPool::operator const VkCommandPool &() const
{
	return m_handle;
}

const VkCommandPool &CommandPool::getCommandPool() const
{
	return m_handle;
}

const std::thread::id &CommandPool::getThreadID() const
{
	return m_thread_id;
}

const QueueUsage &CommandPool::getUsage() const
{
	return m_queue;
}

size_t CommandPool::getHash() const
{
	return m_hash;
}

CommandPool::ResetMode CommandPool::getResetMode() const
{
	return m_reset_mode;
}

void CommandPool::reset()
{
	switch (m_reset_mode)
	{
		case ResetMode::ResetIndividually:
			for (auto &cmd_buffer : m_primary_cmd_buffers)
			{
				cmd_buffer->reset();
			}
			for (auto &cmd_buffer : m_secondary_cmd_buffers)
			{
				cmd_buffer->reset();
			}
			break;
		case ResetMode::ResetPool:
			vkResetCommandPool(GraphicsContext::instance()->getLogicalDevice(), m_handle, 0);
			for (auto &cmd_buffer : m_primary_cmd_buffers)
			{
				cmd_buffer->reset();
			}
			for (auto &cmd_buffer : m_secondary_cmd_buffers)
			{
				cmd_buffer->reset();
			}
			break;
		case ResetMode::AlwaysAllocate:
			m_primary_cmd_buffers.clear();
			m_secondary_cmd_buffers.clear();
			break;
		default:
			throw std::runtime_error("Unknown reset mode for command pools");
	}

	m_active_primary_count   = 0;
	m_active_secondary_count = 0;
}

CommandBuffer &CommandPool::requestCommandBuffer(VkCommandBufferLevel level)
{
	if (level == VK_COMMAND_BUFFER_LEVEL_PRIMARY)
	{
		if (m_active_primary_count < m_primary_cmd_buffers.size())
		{
			return *m_primary_cmd_buffers.at(m_active_primary_count++);
		}

		m_primary_cmd_buffers.emplace_back(std::make_unique<CommandBuffer>(*this, m_queue, level));

		m_active_primary_count++;

		return *m_primary_cmd_buffers.back();
	}
	else
	{
		if (m_active_secondary_count < m_secondary_cmd_buffers.size())
		{
			return *m_secondary_cmd_buffers.at(m_active_secondary_count++);
		}

		m_secondary_cmd_buffers.emplace_back(std::make_unique<CommandBuffer>(*this, m_queue, level));

		m_active_secondary_count++;

		return *m_secondary_cmd_buffers.back();
	}
}
}        // namespace Ilum