#pragma once

#include "RenderFrame.hpp"
#include "Vulkan.hpp"

#include "Pipeline/Pipeline.hpp"
#include "Pipeline/PipelineLayout.hpp"
#include "Pipeline/PipelineState.hpp"

#include "RenderPass/RenderPass.hpp"

#include "Command/CommandBuffer.hpp"
#include "Command/CommandPool.hpp"

namespace Ilum::Graphics
{
class Window;
class Instance;
class PhysicalDevice;
class Surface;
class Device;
class Swapchain;
class PipelineCache;
class DescriptorCache;

class RenderContext
{
  public:
	RenderContext();
	~RenderContext() = default;

	static void NewFrame(VkSemaphore image_ready = VK_NULL_HANDLE);
	static void EndFrame(VkSemaphore render_complete = VK_NULL_HANDLE);

	// ImGui
	static void BeginImGui();
	static void EndImGui();

	// Render Area
	static void              SetRenderArea(const VkExtent2D &extent);
	static const VkExtent2D &GetRenderArea();

	// Descriptor Cache
	static VkDescriptorSet AllocateDescriptorSet(const PipelineState &pso, uint32_t set);
	static void            FreeDescriptorSet(const VkDescriptorSet &descriptor_set);

	// Pipeline Cache
	static Pipeline &      RequestPipeline(const PipelineState &pso);
	static Pipeline &      RequestPipeline(const PipelineState &pso, const RenderPass &render_pass, uint32_t subpass_index = 0);
	static PipelineLayout &RequestPipelineLayout(const PipelineState &pso);

	// One time submit command buffer
	static CommandBuffer &CreateCommandBuffer(QueueFamily queue = QueueFamily::Graphics);
	static void           ResetCommandPool(QueueFamily queue = QueueFamily::Graphics);
	static void           Submit(VkCommandBuffer cmd_buffer, QueueFamily queue_family = QueueFamily::Graphics, uint32_t queue_index = 0, VkFence fence = VK_NULL_HANDLE);
	static void           Submit(const std::vector<VkSubmitInfo> &submit_infos, QueueFamily queue_family = QueueFamily::Graphics, uint32_t queue_index = 0, VkFence fence = VK_NULL_HANDLE);

	// Wait
	static void WaitDevice();
	static void WaitQueue(QueueFamily queue_family = QueueFamily::Graphics, uint32_t queue_index = 0);

	// Set debug name
	template <typename T>
	static void SetName(const T &data, const char *name)
	{
		Graphics::VKDebugger::SetName(*Get().m_device, data, name);
	}

	static Window &        GetWindow();
	static Instance &      GetInstance();
	static Surface &       GetSurface();
	static Device &        GetDevice();
	static PhysicalDevice &GetPhysicalDevice();
	static Swapchain &     GetSwapchain();
	static RenderFrame &   GetFrame();

  private:
	static RenderContext &Get();

	void Recreate();

  private:
	// Device
	std::unique_ptr<Window>          m_window           = nullptr;
	std::unique_ptr<Instance>        m_instance         = nullptr;
	std::unique_ptr<Surface>         m_surface          = nullptr;
	std::unique_ptr<Device>          m_device           = nullptr;
	std::unique_ptr<PhysicalDevice>  m_physical_device  = nullptr;
	std::unique_ptr<Swapchain>       m_swapchain        = nullptr;
	std::unique_ptr<DescriptorCache> m_descriptor_cache = nullptr;
	std::unique_ptr<PipelineCache>   m_pipeline_cache   = nullptr;

	// Frame
	std::vector<std::unique_ptr<RenderFrame>> m_render_frames;
	uint32_t                                  m_active_frame = 0;

	// Command Pool for one time submit
	std::map<size_t, std::unique_ptr<CommandPool>> m_command_pools;

	// Render Area
	VkExtent2D m_render_area = {};
};
}        // namespace Ilum::Graphics