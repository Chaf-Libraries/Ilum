#include "RenderPass.hpp"
#include "RenderTarget.hpp"

#include "Core/Device/LogicalDevice.hpp"
#include "Core/Device/Surface.hpp"
#include "Core/Graphics/GraphicsContext.hpp"

namespace Ilum
{
RenderPass::RenderPass(const RenderTarget &render_target, VkFormat depth_format, VkSampleCountFlagBits samples)
{
	std::vector<VkAttachmentDescription> attachment_descriptions;

	// Create render pass attachment description
	for (const auto &attachment : render_target.getAttachments())
	{
		VkAttachmentDescription attachment_description = {};
		attachment_description.samples                 = attachment.getSamples();
		attachment_description.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachment_description.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
		attachment_description.stencilLoadOp           = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment_description.stencilStoreOp          = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment_description.initialLayout           = VK_IMAGE_LAYOUT_UNDEFINED;

		switch (attachment.getType())
		{
			case Attachment::Type::Image:
				attachment_description.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
				attachment_description.format      = attachment.getFormat();
				break;
			case Attachment::Type::Depth:
				attachment_description.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
				attachment_description.format      = depth_format;
				break;
			case Attachment::Type::Swapchain:
				attachment_description.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
				attachment_description.format      = GraphicsContext::instance()->getSurface().getFormat().format;
				break;
			default:
				break;
		}

		attachment_descriptions.emplace_back(attachment_description);
	}

	// Create subpass and its dependencies
	std::vector<VkSubpassDependency> subpass_dependencies;
	std::vector<VkSubpassDescription> subpass_descriptions;

	for (const auto& subpass : render_target.getSubpasses())
	{
		std::vector<VkAttachmentReference> subpass_color_attachments;
		std::optional<uint32_t>            depth_attachment;

		for (const auto& attachment_binding : subpass.getAttachmentBindings())
		{
			auto attachment = render_target.getAttachment(attachment_binding);

			if (!attachment)
			{
				VK_ERROR("Failed to find a renderpass attachment bound to: {}", attachment_binding);
			}
		}
	}
}

RenderPass::~RenderPass()
{
	vkDestroyRenderPass(GraphicsContext::instance()->getLogicalDevice(), m_handle, nullptr);
}

RenderPass::operator const VkRenderPass &() const
{
	return m_handle;
}

const VkRenderPass &RenderPass::getRenderPass() const
{
	return m_handle;
}
}        // namespace Ilum