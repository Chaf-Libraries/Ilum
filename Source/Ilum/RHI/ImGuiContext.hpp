#pragma once

#include <volk.h>

#include <unordered_map>

namespace Ilum
{
class CommandBuffer;
class Texture;
class Sampler;
class RHIDevice;
class Window;

class ImGuiContext
{
  public:
	ImGuiContext(Window *window, RHIDevice *device);

	~ImGuiContext();

	void BeginFrame();
	void Render(CommandBuffer &cmd_buffer);
	void EndFrame();

	void *TextureID(VkImageView &view, VkSampler &sampler);

  private:
	void Flush();

	void CreateDescriptorPool();
	void CreateRenderPass();
	void CreateFramebuffer();

	void SetStyle();

  private:
	Window    *p_window = nullptr;
	RHIDevice *p_device = nullptr;

	VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
	VkRenderPass     m_render_pass     = VK_NULL_HANDLE;
	std::vector<VkFramebuffer> m_frame_buffers;

	std::unordered_map<size_t, VkDescriptorSet> m_texture_id_mapping;
};
}        // namespace Ilum