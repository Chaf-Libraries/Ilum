#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class RenderTarget;
class CommandBuffer;

class ImGuiContext
{
  public:
	ImGuiContext();

	~ImGuiContext();

	void render(const CommandBuffer& command_buffer);

  private:
	scope<RenderTarget> m_render_target;
	VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
};
}        // namespace Ilum