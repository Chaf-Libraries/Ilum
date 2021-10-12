#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class RenderPass;
class RenderTarget;
class Swapchain;
class ImageDepth;
class Image2D;

class Framebuffer
{
  public:
	Framebuffer(const RenderTarget& render_target, const RenderPass& render_pass);

	~Framebuffer();

	const std::vector<VkFramebuffer> &getFramebuffers() const;

  private:
	std::vector<VkFramebuffer> m_framebuffers;
};
}        // namespace Ilum