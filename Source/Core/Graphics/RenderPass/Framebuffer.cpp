#include "Framebuffer.hpp"
#include "RenderPass.hpp"
#include "RenderTarget.hpp"
#include "Swapchain.hpp"

#include "Core/Graphics/GraphicsContext.hpp"
#include "Core/Graphics/Image/Image2D.hpp"
#include "Core/Graphics/Image/ImageDepth.hpp"
#include "Core/Device/LogicalDevice.hpp"

namespace Ilum
{
Framebuffer::Framebuffer(const RenderTarget &render_target, const RenderPass &render_pass)
{
	m_framebuffers.resize(GraphicsContext::instance()->getSwapchain().getImageCount());

	for (uint32_t i = 0; i < m_framebuffers.size(); i++)
	{
		std::vector<VkImageView> attachments;

		for (const auto &attachment : render_target.getAttachments())
		{
			switch (attachment.getType())
			{
				case Attachment::Type::Image:
					attachments.push_back(render_target.getColorAttachment(attachment.getBinding())->getView());
					break;
				case Attachment::Type::Depth:
					attachments.push_back(render_target.getDepthStencil()->getView());
					break;
				case Attachment::Type::Swapchain:
					attachments.push_back(GraphicsContext::instance()->getSwapchain().getImageViews()[i]);
					break;
				default:
					break;
			}
		}

		VkFramebufferCreateInfo frame_buffer_create_info = {};
		frame_buffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frame_buffer_create_info.renderPass              = render_pass;
		frame_buffer_create_info.attachmentCount         = static_cast<uint32_t>(attachments.size());
		frame_buffer_create_info.pAttachments            = attachments.data();
		frame_buffer_create_info.width                   = static_cast<uint32_t>(render_target.getRenderArea().extent.width);
		frame_buffer_create_info.height                  = static_cast<uint32_t>(render_target.getRenderArea().extent.height);
		frame_buffer_create_info.layers                  = 1;
		vkCreateFramebuffer(GraphicsContext::instance()->getLogicalDevice(), &frame_buffer_create_info, nullptr, &m_framebuffers[i]);
	}
}

Framebuffer::~Framebuffer()
{
	for (const auto& framebuffer : m_framebuffers)
	{
		vkDestroyFramebuffer(GraphicsContext::instance()->getLogicalDevice(), framebuffer, nullptr);
	}
}

const std::vector<VkFramebuffer> &Framebuffer::getFramebuffers() const
{
	return m_framebuffers;
}
}        // namespace Ilum