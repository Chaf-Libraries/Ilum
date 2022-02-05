#pragma once

#include <Core/Event/Event.hpp>

namespace Ilum::Vulkan
{
class Instance;
class Device;
class ShaderCache;
class DescriptorCache;
class PipelineCache;
class Swapchain;
class RenderFrame;

class RenderContext
{
  public:
	RenderContext();

	~RenderContext() = default;

	void OnImGui();
	void OnEvent(const Core::Event &event);

	void NewFrame();
	void EndFrame();

  public:
	static Instance &   GetInstance();
	static Device &     GetDevice();
	static ShaderCache &GetShaderCache();
	static DescriptorCache &GetDescriptorCache();
	static PipelineCache &  GetPipelineCache();
	static Swapchain &  GetSwapchain();
	static RenderFrame &GetFrame();

  private:
	static std::unique_ptr<Instance>                 s_instance;
	static std::unique_ptr<Device>                   s_device;
	static std::unique_ptr<ShaderCache>              s_shader_cache;
	static std::unique_ptr<DescriptorCache>          s_descriptor_cache;
	static std::unique_ptr<PipelineCache>            s_pipeline_cache;
	static std::unique_ptr<Swapchain>                s_swapchain;
	static std::vector<std::unique_ptr<RenderFrame>> s_frames;

	static uint32_t s_active_frame;
};
}        // namespace Ilum::Vulkan