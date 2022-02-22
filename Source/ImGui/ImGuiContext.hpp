#pragma once

#include <Graphics/Command/CommandBuffer.hpp>
#include <Graphics/RenderContext.hpp>

namespace Ilum
{
class ImGuiContext
{
  public:
	ImGuiContext();
	~ImGuiContext();

	static void Initialize();
	static void Destroy();
	static void Recreate();
	static void BeginImGui();
	static void EndImGui();
	static void Render(const Graphics::CommandBuffer &cmd_buffer);

  private:
	static void Flush();
	static void SetStyle();

  private:
	static std::unique_ptr<ImGuiContext> s_instance;

	struct
	{
		VkDescriptorPool                                 descriptor_pool = VK_NULL_HANDLE;
		std::unordered_map<VkImageView, VkDescriptorSet> m_texture_id;
		VkSampler                                        m_sampler = VK_NULL_HANDLE;
		VkRenderPass                                     m_render_pass = VK_NULL_HANDLE;
		// File dialog
		std::unordered_map<VkDescriptorSet, std::unique_ptr<Graphics::Image>> m_filedialog_image_cache;
		std::vector<VkDescriptorSet>                         m_deprecated_descriptor_sets;
	}m_render_data;

};
}        // namespace Ilum