#pragma once

#include <RHI/RHIRenderTarget.hpp>

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

	virtual ~RenderTarget() override;

	virtual RHIRenderTarget &Set(uint32_t slot, RHITexture *texture, RHITextureDimension dimension, const ColorAttachment &attachment) override;
	virtual RHIRenderTarget &Set(uint32_t slot, RHITexture *texture, const TextureRange &range, const ColorAttachment &attachment) override;
	virtual RHIRenderTarget &Set(RHITexture *texture, RHITextureDimension dimension, const DepthStencilAttachment &attachment) override;
	virtual RHIRenderTarget &Set(RHITexture *texture, const TextureRange &range, const DepthStencilAttachment &attachment) override;

	VkRenderPass GetRenderPass() const;

	VkFramebuffer GetFramebuffer() const;

	VkRect2D GetRenderArea() const;

	std::vector<VkClearValue> GetClearValue() const;

	const std::vector<VkRenderingAttachmentInfo> &GetColorAttachments();

	const std::optional<VkRenderingAttachmentInfo> &GetDepthAttachment();

	const std::optional<VkRenderingAttachmentInfo> &GetStencilAttachment();

	const std::vector<VkFormat> &GetColorFormats();

	const std::optional<VkFormat> &GetDepthFormat();

	const std::optional<VkFormat> &GetStencilFormat();

	size_t GetFormatHash() const;

	size_t GetHash() const;

	virtual RHIRenderTarget &Clear() override;

  private:
	// Dynamic Rendering
	std::vector<VkRenderingAttachmentInfo>   m_color_attachments;
	std::optional<VkRenderingAttachmentInfo> m_depth_attachment;
	std::optional<VkRenderingAttachmentInfo> m_stencil_attachment;

	std::vector<VkFormat>   m_color_formats;
	std::optional<VkFormat> m_depth_format;
	std::optional<VkFormat> m_stencil_format;

	size_t m_hash = 0;
};
}        // namespace Ilum::Vulkan