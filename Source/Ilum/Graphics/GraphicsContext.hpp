#pragma once

#include "Engine/Subsystem.hpp"
#include <Core/Event.hpp>
#include "Graphics/Synchronization/QueueSystem.hpp"
#include <Core/Timer.hpp>
#include "Utils/PCH.hpp"

#include <Graphics/Command/CommandBuffer.hpp>

namespace Ilum
{
class DescriptorCache;
class ShaderCache;
class ImGuiContext;
class Profiler;

class GraphicsContext : public TSubsystem<GraphicsContext>
{
  public:
	GraphicsContext(Context *context);

	~GraphicsContext() = default;

	DescriptorCache &getDescriptorCache();

	ShaderCache &getShaderCache();

	QueueSystem &getQueueSystem();

	Profiler &getProfiler();

	const VkPipelineCache &getPipelineCache() const;

	uint32_t getFrameIndex() const;

	const VkSemaphore &getPresentCompleteSemaphore() const;

	const VkSemaphore &getRenderCompleteSemaphore() const;

	uint64_t getFrameCount() const;

	bool isVsync() const;

	void setVsync(bool vsync);

	Graphics::CommandBuffer *getCurrentCommandBuffer() const;

  public:
	virtual bool onInitialize() override;

	virtual void onPreTick() override;

	virtual void onTick(float delta_time) override;

	virtual void onPostTick() override;

	virtual void onShutdown() override;

  public:
	void createSwapchain(bool vsync = false);

	void createCommandBuffer();

  private:
	void newFrame();

	void submitFrame();

  private:
	scope<DescriptorCache> m_descriptor_cache = nullptr;
	scope<ShaderCache>     m_shader_cache     = nullptr;
	scope<Profiler>        m_profiler         = nullptr;

	Graphics::CommandBuffer *m_cmd_buffer = nullptr;

	// Present resource
	std::vector<VkSemaphore>          m_present_complete;
	std::vector<VkSemaphore>          m_render_complete;
	std::vector<VkFence>              m_flight_fences;
	uint32_t                          m_current_frame = 0;
	bool                              m_vsync         = false;

	VkPipelineCache m_pipeline_cache = VK_NULL_HANDLE;

	uint64_t m_frame_count = 0;

	std::mutex m_command_pool_mutex;
	std::mutex m_command_buffer_mutex;

	scope<QueueSystem> m_queue_system = nullptr;

  public:
	Core::Event<> Swapchain_Rebuild_Event;
};
}        // namespace Ilum