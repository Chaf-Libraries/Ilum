#include "Queue.hpp"
#include "Commond.hpp"
#include "Device.hpp"
#include "Synchronization.hpp"

namespace Ilum::Vulkan
{
Queue::Queue(RHIDevice *device, RHIQueueFamily family, uint32_t queue_index) :
    RHIQueue(device, family, queue_index)
{
	vkGetDeviceQueue(static_cast<Device *>(p_device)->GetDevice(), static_cast<Device *>(p_device)->GetQueueFamily(family), queue_index % static_cast<Device *>(p_device)->GetQueueCount(family), &m_handle);
}

void Queue::Submit(const std::vector<RHICommond *> &cmds, const std::vector<RHISemaphore *> &signal_semaphores, const std::vector<RHISemaphore *> &wait_semaphores)
{
	m_submits.emplace_back(std::make_tuple(cmds, signal_semaphores, wait_semaphores));
}

void Queue::Execute(RHIFence *fence)
{
	std::vector<VkSubmitInfo> submit_infos;

	std::vector<std::vector<VkSemaphore>> signal_vk_semaphores;
	std::vector<std::vector<VkSemaphore>> wait_vk_semaphores;
	std::vector<std::vector<VkPipelineStageFlags>> pipeline_stage_flags;

	for (auto &&[cmds, signal_semaphores, wait_semaphores] : m_submits)
	{
		VkSubmitInfo submit_info = {};
		submit_info.sType        = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		signal_vk_semaphores.push_back({});
		std::transform(signal_semaphores.cbegin(), signal_semaphores.cend(), std::back_inserter(signal_vk_semaphores.back()),
		               [](RHISemaphore *semaphore) { return static_cast<Semaphore *>(semaphore)->GetHandle(); });

		wait_vk_semaphores.push_back({});
		std::transform(wait_semaphores.cbegin(), wait_semaphores.cend(), std::back_inserter(wait_vk_semaphores.back()),
		               [](RHISemaphore *semaphore) { return static_cast<Semaphore *>(semaphore)->GetHandle(); });

		pipeline_stage_flags.push_back(std::vector<VkPipelineStageFlags>(wait_vk_semaphores.size()));
		// TODO: Some optimize here
		std::fill(pipeline_stage_flags.back().begin(), pipeline_stage_flags.back().end(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

		submit_info.signalSemaphoreCount = static_cast<uint32_t>(signal_vk_semaphores.back().size());
		submit_info.pSignalSemaphores    = signal_vk_semaphores.back().data();
		submit_info.waitSemaphoreCount   = static_cast<uint32_t>(wait_vk_semaphores.back().size());
		submit_info.pWaitSemaphores      = wait_vk_semaphores.back().data();
		submit_info.pWaitDstStageMask    = pipeline_stage_flags.back().data();
		submit_infos.push_back(submit_info);
	}

	if (!submit_infos.empty())
	{
		vkQueueSubmit(m_handle, static_cast<uint32_t>(submit_infos.size()), submit_infos.data(), fence ? static_cast<Fence *>(fence)->GetHandle() : nullptr);
		m_submits.clear();
	}
}

VkQueue Queue::GetHandle() const
{
	return m_handle;
}
}        // namespace Ilum::Vulkan