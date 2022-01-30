#include "VKCommandBuffer.hpp"
#include "VKCommandPool.hpp"
#include "VKContext.hpp"
#include "VKDevice.hpp"

namespace Ilum::RHI::Vulkan
{
VKCommandBuffer::VKCommandBuffer(const CmdUsage &usage, const std::thread::id &thread_id) :
    RHICommand(usage, thread_id)
{
	auto &cmd_pool = VKContext::GetDevice().AcquireCommandPool(usage, thread_id);
	m_pool_index   = cmd_pool.GetHash();

	VkCommandBufferAllocateInfo allocate_info = {};
	allocate_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocate_info.commandPool                 = cmd_pool;
	allocate_info.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocate_info.commandBufferCount          = 1;

	vkAllocateCommandBuffers(VKContext::GetDevice(), &allocate_info, &m_handle);

	m_state = CommandState::Initial;
}

VKCommandBuffer::~VKCommandBuffer()
{
	if (VKContext::GetDevice().HasCommandPool(m_pool_index)  && m_handle)
	{
		vkFreeCommandBuffers(VKContext::GetDevice(), VKContext::GetDevice().GetCommandPool(m_pool_index), 1, &m_handle);
	}
}

VKCommandBuffer::operator const VkCommandBuffer &() const
{
	return m_handle;
}

const VkCommandBuffer &VKCommandBuffer::GetHandle() const
{
	return m_handle;
}

const CommandState &VKCommandBuffer::GetState() const
{
	return m_state;
}

void VKCommandBuffer::Begin()
{
	VkCommandBufferBeginInfo begin_info = {};
	begin_info.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	begin_info.pInheritanceInfo         = nullptr;

	vkBeginCommandBuffer(m_handle, &begin_info);

	m_state = CommandState::Recording;
}

void VKCommandBuffer::End()
{
	vkEndCommandBuffer(m_handle);

	m_state = CommandState::Executable;
}

void VKCommandBuffer::Reset()
{
	vkResetCommandBuffer(m_handle, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
}
}        // namespace Ilum::RHI::Vulkan

#ifdef USE_VULKAN
std::function<std::shared_ptr<Ilum::RHI::RHICommand>(const Ilum::RHI::CmdUsage &, const std::thread::id &)> Ilum::RHI::RHICommand::CreateFunc =
    [](const Ilum::RHI::CmdUsage &usage, const std::thread::id &thread_id)
    -> std::shared_ptr<Ilum::RHI::RHICommand> { return std::make_shared<Ilum::RHI::Vulkan::VKCommandBuffer>(usage, thread_id); };
#endif        // USE_VULKAN