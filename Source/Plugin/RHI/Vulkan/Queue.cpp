#include "Queue.hpp"
#include "Command.hpp"
#include "Device.hpp"
#include "Synchronization.hpp"

namespace Ilum::Vulkan
{
Queue::Queue(RHIDevice *device) :
    RHIQueue(device), p_device(static_cast<Device *>(device))
{
	for (uint32_t i = 0; i < p_device->GetQueueCount(RHIQueueFamily::Graphics); i++)
	{
		m_queues[RHIQueueFamily::Graphics].push_back(VK_NULL_HANDLE);
		vkGetDeviceQueue(p_device->GetDevice(), p_device->GetQueueFamily(RHIQueueFamily::Graphics), i, &m_queues[RHIQueueFamily::Graphics].back());
		m_queue_fences.emplace(m_queues[RHIQueueFamily::Graphics].back(), VK_NULL_HANDLE);

		{
			std::string queue_name = "Graphics Queue - " + std::to_string(i);

			VkDebugUtilsObjectNameInfoEXT info = {};
			info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
			info.pObjectName                   = queue_name.c_str();
			info.objectHandle                  = (uint64_t) m_queues[RHIQueueFamily::Graphics].back();
			info.objectType                    = VK_OBJECT_TYPE_QUEUE;
			static_cast<Device *>(p_device)->SetVulkanObjectName(info);
		}
	}

	for (uint32_t i = 0; i < p_device->GetQueueCount(RHIQueueFamily::Transfer); i++)
	{
		m_queues[RHIQueueFamily::Transfer].push_back(VK_NULL_HANDLE);
		vkGetDeviceQueue(p_device->GetDevice(), p_device->GetQueueFamily(RHIQueueFamily::Transfer), i, &m_queues[RHIQueueFamily::Transfer].back());
		m_queue_fences.emplace(m_queues[RHIQueueFamily::Transfer].back(), VK_NULL_HANDLE);

		{
			std::string queue_name = "Transfer Queue - " + std::to_string(i);

			VkDebugUtilsObjectNameInfoEXT info = {};
			info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
			info.pObjectName                   = queue_name.c_str();
			info.objectHandle                  = (uint64_t) m_queues[RHIQueueFamily::Graphics].back();
			info.objectType                    = VK_OBJECT_TYPE_QUEUE;
			static_cast<Device *>(p_device)->SetVulkanObjectName(info);
		}
	}

	for (uint32_t i = 0; i < p_device->GetQueueCount(RHIQueueFamily::Compute); i++)
	{
		m_queues[RHIQueueFamily::Compute].push_back(VK_NULL_HANDLE);
		vkGetDeviceQueue(p_device->GetDevice(), p_device->GetQueueFamily(RHIQueueFamily::Compute), i, &m_queues[RHIQueueFamily::Compute].back());
		m_queue_fences.emplace(m_queues[RHIQueueFamily::Compute].back(), VK_NULL_HANDLE);

		{
			std::string queue_name = "Compute Queue - " + std::to_string(i);

			VkDebugUtilsObjectNameInfoEXT info = {};
			info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
			info.pObjectName                   = queue_name.c_str();
			info.objectHandle                  = (uint64_t) m_queues[RHIQueueFamily::Graphics].back();
			info.objectType                    = VK_OBJECT_TYPE_QUEUE;
			static_cast<Device *>(p_device)->SetVulkanObjectName(info);
		}
	}

	for (auto &[queue, fence] : m_queue_fences)
	{
		VkFenceCreateInfo create_info = {};
		create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		vkCreateFence(p_device->GetDevice(), &create_info, nullptr, &fence);
	}

	m_queue_index[RHIQueueFamily::Graphics] = 0;
	m_queue_index[RHIQueueFamily::Transfer] = 0;
	m_queue_index[RHIQueueFamily::Compute]  = 0;
}

Queue::~Queue()
{
	p_device->WaitIdle();

	for (auto &[queue, fence] : m_queue_fences)
	{
		vkDestroyFence(p_device->GetDevice(), fence, nullptr);
	}
	m_queue_fences.clear();
}

void Queue::Execute(RHIQueueFamily family, const std::vector<SubmitInfo> &submit_infos, RHIFence *fence)
{
	VkQueue queue    = m_queues.at(family).at(m_queue_index[family]++);
	VkFence vk_fence = fence ? static_cast<Fence *>(fence)->GetHandle() : m_queue_fences.at(queue);

	if (vkGetFenceStatus(p_device->GetDevice(), vk_fence) == VK_SUCCESS)
	{
		vkWaitForFences(p_device->GetDevice(), 1, &vk_fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
		vkResetFences(p_device->GetDevice(), 1, &vk_fence);
	}

	std::vector<VkSubmitInfo> vk_submit_infos;
	vk_submit_infos.reserve(submit_infos.size());

	std::vector<std::vector<VkPipelineStageFlags>> pipeline_stage_flags(submit_infos.size());
	std::vector<std::vector<VkCommandBuffer>>      cmd_buffers(submit_infos.size());
	std::vector<std::vector<VkSemaphore>>          wait_semaphores(submit_infos.size());
	std::vector<std::vector<VkSemaphore>>          signal_semaphores(submit_infos.size());

	for (uint32_t i = 0; i < submit_infos.size(); i++)
	{
		const auto &submit_info = submit_infos[i];

		pipeline_stage_flags[i].resize(submit_info.wait_semaphores.size());
		std::fill(pipeline_stage_flags[i].begin(), pipeline_stage_flags[i].end(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

		cmd_buffers[i].reserve(submit_info.cmd_buffers.size());
		for (auto &cmd_buffer : submit_info.cmd_buffers)
		{
			cmd_buffers[i].push_back(static_cast<Command *>(cmd_buffer)->GetHandle());
		}

		wait_semaphores[i].reserve(submit_info.wait_semaphores.size());
		for (auto &wait_semaphore : submit_info.wait_semaphores)
		{
			wait_semaphores[i].push_back(static_cast<Semaphore *>(wait_semaphore)->GetHandle());
		}

		signal_semaphores[i].reserve(submit_info.signal_semaphores.size());
		for (auto &signal_semaphore : submit_info.signal_semaphores)
		{
			signal_semaphores[i].push_back(static_cast<Semaphore *>(signal_semaphore)->GetHandle());
		}

		VkSubmitInfo vk_submit_info = {};
		vk_submit_info.sType        = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		vk_submit_info.commandBufferCount   = static_cast<uint32_t>(cmd_buffers[i].size());
		vk_submit_info.pCommandBuffers      = cmd_buffers[i].data();
		vk_submit_info.signalSemaphoreCount = static_cast<uint32_t>(signal_semaphores[i].size());
		vk_submit_info.pSignalSemaphores    = signal_semaphores[i].data();
		vk_submit_info.waitSemaphoreCount   = static_cast<uint32_t>(wait_semaphores[i].size());
		vk_submit_info.pWaitSemaphores      = wait_semaphores[i].data();
		vk_submit_info.pWaitDstStageMask    = pipeline_stage_flags[i].data();

		vk_submit_infos.push_back(std::move(vk_submit_info));
	}

	vkQueueSubmit(queue, static_cast<uint32_t>(vk_submit_infos.size()), vk_submit_infos.data(), vk_fence);

	m_queue_index[family] = m_queue_index[family] % p_device->GetQueueCount(family);
}

void Queue::Execute(RHICommand *cmd_buffer)
{
	auto vk_cmd_buffer = static_cast<Command *>(cmd_buffer)->GetHandle();

	RHIQueueFamily family = cmd_buffer->GetQueueFamily();

	size_t  index = m_queue_index[family];
	VkQueue queue = m_queues.at(family).at(index);
	VkFence fence = m_queue_fences.at(queue);

	if (vkGetFenceStatus(p_device->GetDevice(), fence) == VK_SUCCESS)
	{
		vkWaitForFences(p_device->GetDevice(), 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
		vkResetFences(p_device->GetDevice(), 1, &fence);
	}

	VkSubmitInfo submit_info         = {};
	submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount   = 1;
	submit_info.pCommandBuffers      = &vk_cmd_buffer;
	submit_info.signalSemaphoreCount = 0;
	submit_info.pSignalSemaphores    = nullptr;
	submit_info.waitSemaphoreCount   = 0;
	submit_info.pWaitSemaphores      = nullptr;
	submit_info.pWaitDstStageMask    = nullptr;

	vkQueueSubmit(m_queues[family][index], 1, &submit_info, fence);
	vkWaitForFences(p_device->GetDevice(), 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
	vkResetFences(p_device->GetDevice(), 1, &fence);
}

void Queue::Wait()
{
	std::vector<VkFence> fences;
	fences.reserve(m_queue_fences.size());

	for (auto &[queue, fence] : m_queue_fences)
	{
		if (vkGetFenceStatus(p_device->GetDevice(), fence) == VK_SUCCESS)
		{
			fences.push_back(fence);
		}
	}

	vkWaitForFences(p_device->GetDevice(), static_cast<uint32_t>(fences.size()), fences.data(), VK_TRUE, std::numeric_limits<uint64_t>::max());
	vkResetFences(p_device->GetDevice(), static_cast<uint32_t>(fences.size()), fences.data());
}

VkQueue Queue::GetHandle(RHIQueueFamily family, uint32_t index) const
{
	return m_queues.at(family).at(index % m_queues.at(family).size());
}
}        // namespace Ilum::Vulkan