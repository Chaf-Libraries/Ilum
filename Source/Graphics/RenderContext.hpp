#pragma once

#include "Vulkan.hpp"

namespace Ilum::Graphics
{
class Window;
class Instance;
class PhysicalDevice;
class Surface;
class Device;
class Swapchain;
class RenderFrame;

class RenderContext
{
  public:
	RenderContext();
	~RenderContext() = default;

	static void NewFrame(VkSemaphore image_ready = VK_NULL_HANDLE);
	static void EndFrame(VkSemaphore render_complete = VK_NULL_HANDLE);

	static void BeginImGui();
	static void EndImGui();

	static void              SetRenderArea(const VkExtent2D &extent);
	static const VkExtent2D &GetRenderArea();

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
	std::unique_ptr<Window>         m_window          = nullptr;
	std::unique_ptr<Instance>       m_instance        = nullptr;
	std::unique_ptr<Surface>        m_surface         = nullptr;
	std::unique_ptr<Device>         m_device          = nullptr;
	std::unique_ptr<PhysicalDevice> m_physical_device = nullptr;
	std::unique_ptr<Swapchain>      m_swapchain       = nullptr;

	// Frame
	std::vector<std::unique_ptr<RenderFrame>> m_render_frames;
	uint32_t                                  m_active_frame = 0;

	// Render Area
	VkExtent2D m_render_area = {};
};
}        // namespace Ilum::Graphics