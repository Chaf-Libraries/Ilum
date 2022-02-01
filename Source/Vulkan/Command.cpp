#include "Command.hpp"
#include "Device.hpp"
#include "RenderContext.hpp"
#include "Synchronize.hpp"

#include <Core/Hash.hpp>

namespace Ilum::Vulkan
{
CommandPool::CommandPool(QueueFamily queue, ResetMode reset_mode, const std::thread::id &thread_id) :
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

CommandPool ::~CommandPool()
{
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
			for (auto &cmd_buffer : m_cmd_buffers)
			{
				cmd_buffer->Reset();
			}
			break;
		case ResetMode::ResetPool:
			vkResetCommandPool(RenderContext::GetDevice(), m_handle, 0);
			for (auto &cmd_buffer : m_cmd_buffers)
			{
				cmd_buffer->Reset();
			}
			break;
		case ResetMode::AlwaysAllocate:
			m_cmd_buffers.clear();
			break;
		default:
			throw std::runtime_error("Unknown reset mode for command pools");
	}

	m_active_count = 0;
}

CommandBuffer &CommandPool::RequestCommandBuffer(VkCommandBufferLevel level)
{
	if (m_active_count < m_cmd_buffers.size())
	{
		return *m_cmd_buffers.at(m_active_count++);
	}

	m_cmd_buffers.emplace_back(std::make_unique<CommandBuffer>(*this, level));

	m_active_count++;

	return *m_cmd_buffers.back();
}

CommandBuffer::CommandBuffer(const CommandPool &cmd_pool, VkCommandBufferLevel level) :
    m_level(level), m_pool(cmd_pool)
{
	VkCommandBufferAllocateInfo allocate_info = {};
	allocate_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocate_info.commandPool                 = cmd_pool;
	allocate_info.level                       = level;
	allocate_info.commandBufferCount          = 1;

	vkAllocateCommandBuffers(RenderContext::GetDevice(), &allocate_info, &m_handle);

	m_state = CommandBuffer::State::Initial;
}

CommandBuffer::~CommandBuffer()
{
	if (m_handle)
	{
		vkFreeCommandBuffers(RenderContext::GetDevice(), m_pool, 1, &m_handle);
	}
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
	//assert(m_state == State::Invalid);

	if (m_pool.GetResetMode() == CommandPool::ResetMode::ResetIndividually)
	{
		vkResetCommandBuffer(m_handle, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
	}

	m_state = State::Initial;
}

void CommandBuffer::Begin()
{
	assert(m_state == State::Initial);

	VkCommandBufferBeginInfo begin_info = {};
	begin_info.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	begin_info.pInheritanceInfo         = nullptr;

	vkBeginCommandBuffer(m_handle, &begin_info);

	m_state = State::Recording;
}

void CommandBuffer::End()
{
	assert(m_state == State::Recording);

	vkEndCommandBuffer(m_handle);

	m_state = State::Executable;
}

void CommandBuffer::SubmitIdle()
{
	assert(m_state == State::Executable);

	VkSubmitInfo submit_info       = {};
	submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &m_handle;

	VkFenceCreateInfo fence_create_info = {};
	fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

	Fence fence;

	fence.Reset();

	if (!VK_CHECK(vkQueueSubmit(RenderContext::GetDevice().GetQueue(m_queue), 1, &submit_info, fence)))
	{
		LOG_ERROR("Failed to submit queue!");
		return;
	}

	fence.Wait();

	m_state = State::Invalid;
}
}        // namespace Ilum::Vulkan