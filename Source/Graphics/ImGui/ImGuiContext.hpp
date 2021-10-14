#pragma once

#include "Utils/PCH.hpp"

#include "Engine/Subsystem.hpp"

namespace Ilum
{
class CommandBuffer;
class Image;
class Sampler;

class ImGuiContext : public TSubsystem<ImGuiContext>
{
  public:
	ImGuiContext(Context *context = nullptr);

	~ImGuiContext() = default;

	virtual bool onInitialize() override;

	virtual void onPreTick() override;

	virtual void onPostTick() override;

	virtual void onShutdown() override;

	void *textureID(const Image& image, const Sampler& sampler);

	void setDockingSpace(bool enable);

  private:
	void config();

	void uploadFontsData();

  private:
	VkDescriptorPool                                                   m_descriptor_pool = VK_NULL_HANDLE;
	std::array<VkSemaphore, 3>                                         m_render_complete;
	std::array<scope<CommandBuffer>, 3>                                m_command_buffers;
	std::unordered_map<size_t, VkDescriptorSet> m_texture_id_mapping;
	bool                                                               m_dockspace_enable = true;
};
}        // namespace Ilum