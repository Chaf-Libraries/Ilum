#include "Queue.hpp"
#include "Command.hpp"
#include "Device.hpp"
#include "Synchronization.hpp"

namespace Ilum::Vulkan
{
Queue::Queue(RHIDevice *device, RHIQueueFamily family, uint32_t queue_index) :
    RHIQueue(device, family, queue_index)
{
	vkGetDeviceQueue(static_cast<Device *>(p_device)->GetDevice(), static_cast<Device *>(p_device)->GetQueueFamily(family), queue_index % static_cast<Device *>(p_device)->GetQueueCount(family), &m_handle);
}

void Queue::Wait()
{
	vkQueueWaitIdle(m_handle);
}

void Queue::Submit(const std::vector<RHICommand *> &cmds, const std::vector<RHISemaphore *> &signal_semaphores, const std::vector<RHISemaphore *> &wait_semaphores)
{
	m_cmds.reserve(cmds.size());
	m_signal_semaphores.reserve(signal_semaphores.size());
	m_wait_semaphores.reserve(wait_semaphores.size());

	for (auto &cmd : cmds)
	{
		static_cast<Command *>(cmd)->SetState(CommandState::Pending);
		m_cmds.push_back(static_cast<Command *>(cmd)->GetHandle());
	}
	for (auto &signal_semaphore : signal_semaphores)
	{
		m_signal_semaphores.push_back(static_cast<Semaphore *>(signal_semaphore)->GetHandle());
	}
	for (auto &wait_semaphore : wait_semaphores)
	{
		m_wait_semaphores.push_back(static_cast<Semaphore *>(wait_semaphore)->GetHandle());
	}
}

void Queue::Execute(RHIFence *fence)
{
	if (m_cmds.empty() && m_signal_semaphores.empty() && m_wait_semaphores.empty())
	{
		return;
	}

	std::vector<VkPipelineStageFlags> pipeline_stage_flags;

	pipeline_stage_flags.resize(m_wait_semaphores.size());
	std::fill(pipeline_stage_flags.begin(), pipeline_stage_flags.end(), VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

	VkSubmitInfo submit_info = {};
	submit_info.sType        = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	submit_info.commandBufferCount   = static_cast<uint32_t>(m_cmds.size());
	submit_info.pCommandBuffers      = m_cmds.data();
	submit_info.signalSemaphoreCount = static_cast<uint32_t>(m_signal_semaphores.size());
	submit_info.pSignalSemaphores    = m_signal_semaphores.data();
	submit_info.waitSemaphoreCount   = static_cast<uint32_t>(m_wait_semaphores.size());
	submit_info.pWaitSemaphores      = m_wait_semaphores.data();
	submit_info.pWaitDstStageMask    = pipeline_stage_flags.data();

	vkQueueSubmit(m_handle, 1, &submit_info, fence ? static_cast<Fence *>(fence)->GetHandle() : nullptr);

	m_cmds.clear();
	m_signal_semaphores.clear();
	m_wait_semaphores.clear();
}

VkQueue Queue::GetHandle() const
{
	return m_handle;
}
}        // namespace Ilum::Vulkan