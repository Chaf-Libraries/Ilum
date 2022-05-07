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
	void Render();
	void EndFrame();

	void OpenFileDialog(const std::string &key, const std::string &title, const std::string &filter);
	void GetFileDialogResult(const std::string &key, std::function<void(const std::string &)> &&callback);

	void *TextureID(VkImageView view);

  private:
	void Flush();

	void CreateDescriptorPool();
	void CreateRenderPass();
	void CreateFramebuffer();
	void CreateSampler();

	void SetStyle();

  private:
	Window    *p_window = nullptr;
	RHIDevice *p_device = nullptr;

	bool m_destroy = false;

	VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
	VkRenderPass     m_render_pass     = VK_NULL_HANDLE;
	VkSampler        m_sampler         = VK_NULL_HANDLE;

	std::vector<VkFramebuffer> m_frame_buffers;

	std::unordered_map<size_t, VkDescriptorSet> m_texture_id_mapping;

	std::unordered_map<void *, std::unique_ptr<Texture>> m_filedialog_textures;
	std::vector<void *>                                  m_deprecated_filedialog_id;
};
}        // namespace Ilum