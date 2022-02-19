#pragma once

#include "Graphics/Vulkan.hpp"
#include "Resource/Image.hpp"

namespace Ilum::Graphics
{
class RenderPass;
class Device;

class Framebuffer
{
  public:
	Framebuffer(const Device& device, const std::vector<ImageReference> &images, const RenderPass &render_pass, uint32_t layers = 1);
	~Framebuffer();

	Framebuffer(const Framebuffer &) = delete;
	Framebuffer &operator=(const Framebuffer &) = delete;
	Framebuffer(Framebuffer &&)                 = delete;
	Framebuffer &operator=(Framebuffer &&) = delete;

	operator const VkFramebuffer &() const;

	const VkFramebuffer &GetHandle() const;

  private:
	const Device &m_device;
	VkFramebuffer m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum::Graphics