#pragma once

#include "Utils/PCH.hpp"

#include "Engine/Subsystem.hpp"

#include <Graphics/Resource/Image.hpp>

namespace Ilum
{
namespace Graphics
{
class CommandBuffer;
};

namespace Graphics
{
class Sampler;
};

class ImGuiContext
{
  public:
	enum class StyleType
	{
		DarkMode
	};

  public:
	ImGuiContext();

	~ImGuiContext() = default;

	static void initialize();

	static void destroy();

	static void createResouce();

	static void releaseResource();

	static void begin();

	static void render(const Graphics::CommandBuffer &command_buffer);

	static void end();

	static void beginDockingSpace();

	static void endDockingSpace();

	static void *textureID(const Graphics::Image &image, const Graphics::Sampler &sampler);

	static void *textureID(const VkImageView &view, const Graphics::Sampler &sampler);

	static void flush();

	bool enable() const;

	static bool needUpdate();

  private:
	static void setDarkMode();

  private:
	static scope<ImGuiContext> s_instance;

	VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;

	std::unordered_map<size_t, VkDescriptorSet> m_texture_id_mapping;

	// File dialog
	std::unordered_map<VkDescriptorSet, Graphics::Image> m_filedialog_image_cache;
	std::vector<VkDescriptorSet>                         m_deprecated_descriptor_sets;

	// ImGui vertex & index
	int32_t m_vertex_count = 0;
	int32_t m_index_count  = 0;

	bool m_need_update = false;

	static bool s_enable;
};
}        // namespace Ilum