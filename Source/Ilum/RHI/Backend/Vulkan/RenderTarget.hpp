#pragma once

#include "RHIRenderTarget.hpp"

#include <volk.h>

#include <map>

namespace Ilum::Vulkan
{
struct FrameBufferResolve
{
	VkImageView  view;
	VkClearValue clear_value;
};

class RenderTarget : public RHIRenderTarget
{
  public:
	RenderTarget(RHIDevice *device);

	virtual ~RenderTarget() override = default;

	virtual RHIRenderTarget &Add(RHITexture *texture, RHITextureDimension dimension, const ColorAttachment &attachment) override;
	virtual RHIRenderTarget &Add(RHITexture *texture, const TextureRange &range, const ColorAttachment &attachment) override;
	virtual RHIRenderTarget &Add(RHITexture *texture, RHITextureDimension dimension, const DepthStencilAttachment &attachment) override;
	virtual RHIRenderTarget &Add(RHITexture *texture, const TextureRange &range, const DepthStencilAttachment &attachment) override;

	VkRenderPass GetRenderPass() const;

	VkFramebuffer GetFramebuffer() const;

	VkRect2D GetRenderArea() const;

	std::vector<VkClearValue> GetClearValue() const;

  private:
	std::vector<VkAttachmentDescription> m_descriptions;
	std::vector<VkAttachmentReference>   m_color_reference;
	std::optional<VkAttachmentReference> m_depth_stencil_reference;

	std::vector<FrameBufferResolve> m_framebuffer_resolves;

	size_t m_render_pass_hash = 0;
	size_t m_framebuffer_hash = 0;
};
}        // namespace Ilum::Vulkan