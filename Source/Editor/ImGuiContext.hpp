#pragma once

#include "Graphics/Vulkan.hpp"

namespace Ilum::Graphics
{
class Image;

class ImGuiContext
{
  public:
	ImGuiContext();
	~ImGuiContext() = default;

	static void Initialize();
	static void Shutdown();

	static void RecreateResource();

	static void Begin();
	static void End();
	static void Render(VkCommandBuffer cmd_buffer);

  private:
	static ImGuiContext &Get();

	static void Reflesh();

  private:
	static std::unique_ptr<ImGuiContext> s_instance;

	struct
	{
		VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
		VkSampler        sampler         = VK_NULL_HANDLE;

		std::unordered_map<VkImageView, VkDescriptorSet> texture_id;

		std::unordered_map<VkDescriptorSet, std::unique_ptr<Graphics::Image>> filedialog_image_cache;
		std::vector<VkDescriptorSet>                         deprecated_filedialog_descriptor;
	} m_imgui_data;
};
}        // namespace Ilum::Graphics