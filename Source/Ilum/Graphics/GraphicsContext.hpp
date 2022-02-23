#pragma once

#include "Engine/Subsystem.hpp"

#include "Eventing/Event.hpp"

#include "Graphics/Synchronization/QueueSystem.hpp"
#include "Graphics/Command/CommandPool.hpp"

#include "Timing/Stopwatch.hpp"

#include "Utils/PCH.hpp"

namespace Ilum
{
class Instance;
class PhysicalDevice;
class Surface;
class LogicalDevice;
class Swapchain;
class CommandBuffer;
class DescriptorCache;
class ShaderCache;
class ImGuiContext;
class Profiler;
class RenderFrame;

class GraphicsContext : public TSubsystem<GraphicsContext>
{
  public:
	GraphicsContext(Context *context);

	~GraphicsContext() = default;

	const Instance &getInstance() const;

	const PhysicalDevice &getPhysicalDevice() const;

	const Surface &getSurface() const;

	const LogicalDevice &getLogicalDevice() const;

	const Swapchain &getSwapchain() const;

	DescriptorCache &getDescriptorCache();

	ShaderCache &getShaderCache();

	QueueSystem &getQueueSystem();

	Profiler &getProfiler();

	const VkPipelineCache &getPipelineCache() const;

	CommandPool &getCommandPool(QueueUsage usage = QueueUsage::Graphics, CommandPool::ResetMode reset_mode = CommandPool::ResetMode::ResetPool);

	uint32_t getFrameIndex() const;

	const VkSemaphore &getPresentCompleteSemaphore() const;

	const VkSemaphore &getRenderCompleteSemaphore() const;

	void submitCommandBuffer(VkCommandBuffer cmd_buffer);

	RenderFrame &getFrame();

	uint64_t getFrameCount() const;

	bool isVsync() const;

	void setVsync(bool vsync);

  public:
	virtual bool onInitialize() override;

	virtual void onPreTick() override;

	virtual void onTick(float delta_time) override;

	virtual void onPostTick() override;

	virtual void onShutdown() override;

  public:
	void createSwapchain(bool vsync = false);

  private:
	void newFrame();

	void submitFrame();

  private:
	scope<Instance>       m_instance        = nullptr;
	scope<PhysicalDevice> m_physical_device = nullptr;
	scope<Surface>        m_surface         = nullptr;
	scope<LogicalDevice>  m_logical_device  = nullptr;
	scope<Swapchain>      m_swapchain       = nullptr;

	scope<DescriptorCache> m_descriptor_cache = nullptr;
	scope<ShaderCache>     m_shader_cache     = nullptr;
	scope<Profiler>        m_profiler         = nullptr;

	// Command pool per thread
	std::vector<scope<RenderFrame>> m_render_frames;

	std::vector<VkCommandBuffer> m_submit_cmd_buffers;

	// Present resource
	CommandBuffer *          cmd_buffer = nullptr;
	std::vector<VkSemaphore> m_present_complete;
	std::vector<VkSemaphore> m_render_complete;
	uint32_t                 m_current_frame = 0;
	bool                     m_vsync         = false;

	VkPipelineCache m_pipeline_cache = VK_NULL_HANDLE;

	uint64_t m_frame_count = 0;

	std::mutex m_command_pool_mutex;
	std::mutex m_command_buffer_mutex;

	scope<QueueSystem> m_queue_system = nullptr;

  public:
	Event<> Swapchain_Rebuild_Event;
};
}        // namespace Ilum