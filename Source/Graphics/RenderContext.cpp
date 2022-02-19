#include "RenderContext.hpp"
#include "RenderFrame.hpp"

#include "Device/Device.hpp"
#include "Device/Instance.hpp"
#include "Device/PhysicalDevice.hpp"
#include "Device/Surface.hpp"
#include "Device/Swapchain.hpp"
#include "Device/Window.hpp"

#include "Descriptor/DescriptorCache.hpp"
#include "Descriptor/DescriptorPool.hpp"
#include "Descriptor/DescriptorSetLayout.hpp"

#include "Pipeline/PipelineCache.hpp"

#include "Synchronize/FencePool.hpp"
#include "Synchronize/SemaphorePool.hpp"

#include <Core/Hash.hpp>

namespace Ilum::Graphics
{
RenderContext::RenderContext()
{
	m_window          = std::make_unique<Window>();
	m_instance        = std::make_unique<Instance>();
	m_physical_device = std::make_unique<PhysicalDevice>(*m_instance);
	m_surface         = std::make_unique<Surface>(*m_instance, *m_physical_device, m_window->GetHandle());
	m_device          = std::make_unique<Device>(*m_instance, *m_physical_device, *m_surface);
	m_swapchain       = std::make_unique<Swapchain>(VkExtent2D{m_window->GetWidth(), m_window->GetHeight()}, *m_device, *m_surface, *m_physical_device);

	for (uint32_t i = 0; i < m_swapchain->GetImageCount(); i++)
	{
		m_render_frames.emplace_back(std::make_unique<RenderFrame>(*m_device));
	}

	m_pipeline_cache   = std::make_unique<PipelineCache>(*m_device);
	m_descriptor_cache = std::make_unique<DescriptorCache>(*m_device);
}

void RenderContext::NewFrame(VkSemaphore image_ready)
{
	auto result = Get().m_swapchain->AcquireNextImage(image_ready);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
	{
		Get().Recreate();
	}
	GetFrame().Reset();
}

void RenderContext::EndFrame(VkSemaphore render_complete)
{
	auto result = Get().m_swapchain->Present(render_complete);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
	{
		Get().Recreate();
	}
	Get().m_active_frame = (Get().m_active_frame + 1) % Get().m_render_frames.size();
}

void RenderContext::BeginImGui()
{
}

void RenderContext::EndImGui()
{
}

void RenderContext::SetRenderArea(const VkExtent2D &extent)
{
	Get().m_render_area = extent;
}

const VkExtent2D &RenderContext::GetRenderArea()
{
	return Get().m_render_area;
}

VkDescriptorSet RenderContext::AllocateDescriptorSet(const PipelineState &pso, uint32_t set)
{
	return Get().m_descriptor_cache->AllocateDescriptorSet(pso.GetReflectionData(), set);
}

void RenderContext::FreeDescriptorSet(const VkDescriptorSet &descriptor_set)
{
	Get().m_descriptor_cache->Free(descriptor_set);
}

Pipeline &RenderContext::RequestPipeline(const PipelineState &pso)
{
	return Get().m_pipeline_cache->RequestPipeline(pso, RequestPipelineLayout(pso));
}

Pipeline &RenderContext::RequestPipeline(const PipelineState &pso, const RenderPass &render_pass, uint32_t subpass_index)
{
	return Get().m_pipeline_cache->RequestPipeline(pso, RequestPipelineLayout(pso), render_pass, subpass_index);
}

PipelineLayout &RenderContext::RequestPipelineLayout(const PipelineState &pso)
{
	std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
	for (auto &set : pso.GetReflectionData().sets)
	{
		descriptor_set_layouts.push_back(Get().m_descriptor_cache->RequestDescriptorSetLayout(pso.GetReflectionData(), set));
	}

	return Get().m_pipeline_cache->RequestPipelineLayout(pso.GetReflectionData(), descriptor_set_layouts);
}

CommandBuffer &RenderContext::CreateCommandBuffer(QueueFamily queue)
{
	// Request command pool
	auto thread_id = std::this_thread::get_id();

	size_t hash = 0;
	Core::HashCombine(hash, static_cast<size_t>(queue));
	Core::HashCombine(hash, thread_id);

	if (Get().m_command_pools.find(hash) == Get().m_command_pools.end())
	{
		Get().m_command_pools.emplace(hash, std::make_unique<CommandPool>(*Get().m_device, queue, CommandPool::ResetMode::ResetPool, thread_id));
	}

	return Get().m_command_pools.at(hash)->RequestCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
}

void RenderContext::ResetCommandPool(QueueFamily queue)
{
	auto thread_id = std::this_thread::get_id();

	size_t hash = 0;
	Core::HashCombine(hash, static_cast<size_t>(queue));
	Core::HashCombine(hash, thread_id);

	if (Get().m_command_pools.find(hash) != Get().m_command_pools.end())
	{
		Get().m_command_pools.at(hash)->Reset();
	}
}

void RenderContext::Submit(VkCommandBuffer cmd_buffer, QueueFamily queue_family, uint32_t queue_index, VkFence fence)
{
	VkSubmitInfo submit_info       = {};
	submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &cmd_buffer;
	vkQueueSubmit(Get().m_device->GetQueue(queue_family, queue_index), 1, &submit_info, fence);
}

void RenderContext::Submit(const std::vector<VkSubmitInfo> &submit_infos, QueueFamily queue_family, uint32_t queue_index, VkFence fence)
{
	vkQueueSubmit(Get().m_device->GetQueue(queue_family, queue_index), static_cast<uint32_t>(submit_infos.size()), submit_infos.data(), fence);
}

void RenderContext::WaitDevice()
{
	vkDeviceWaitIdle(*Get().m_device);
}

void RenderContext::WaitQueue(QueueFamily queue_family, uint32_t queue_index)
{
	vkQueueWaitIdle(Get().m_device->GetQueue(queue_family, queue_index));
}

Window &RenderContext::GetWindow()
{
	return *Get().m_window;
}

Instance &RenderContext::GetInstance()
{
	return *Get().m_instance;
}

Surface &RenderContext::GetSurface()
{
	return *Get().m_surface;
}

Device &RenderContext::GetDevice()
{
	return *Get().m_device;
}

PhysicalDevice &RenderContext::GetPhysicalDevice()
{
	return *Get().m_physical_device;
}

Swapchain &RenderContext::GetSwapchain()
{
	return *Get().m_swapchain;
}

RenderFrame &RenderContext::GetFrame()
{
	return *Get().m_render_frames[Get().m_active_frame];
}

RenderContext &RenderContext::Get()
{
	static RenderContext render_context;
	return render_context;
}

void RenderContext::Recreate()
{
	WaitDevice();

	m_active_frame = 0;
	m_render_frames.clear();
	m_swapchain = std::make_unique<Swapchain>(VkExtent2D{m_window->GetWidth(), m_window->GetHeight()}, *m_device, *m_surface, *m_physical_device, *m_swapchain);

	for (uint32_t i = 0; i < m_swapchain->GetImageCount(); i++)
	{
		m_render_frames.emplace_back(std::make_unique<RenderFrame>(*m_device));
	}
}
}        // namespace Ilum::Graphics