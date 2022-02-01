#pragma once

#include <Core/Event/Event.hpp>

namespace Ilum::Vulkan
{
class Instance;
class Device;
class Swapchain;
class RenderFrame;

class RenderContext
{
  public:
	RenderContext();

	~RenderContext();

	void OnImGui();
	void OnEvent(const Core::Event &event);

	void NewFrame();
	void EndFrame();

  public:
	static Instance & GetInstance();
	static Device &   GetDevice();
	static Swapchain &GetSwapchain();
	static RenderFrame &GetFrame();

  private:
	static std::unique_ptr<Instance> s_instance;
	static std::unique_ptr<Device>   s_device;
	static std::unique_ptr<Swapchain> s_swapchain;
	static std::vector<std::unique_ptr<RenderFrame>> s_frames;

	static uint32_t s_active_frame;
};
}        // namespace Ilum::Vulkan