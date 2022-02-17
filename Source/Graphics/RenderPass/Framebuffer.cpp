#include "Framebuffer.hpp"
#include "RenderPass.hpp"
#include "Device/Device.hpp"

namespace Ilum::Graphics
{
Framebuffer::Framebuffer(const Device &device, const std::vector<ImageReference> &images, const RenderPass &render_pass, uint32_t layers) :
    m_device(device)
{
	uint32_t                 width = 0, height = 0;
	std::vector<VkImageView> views;
	for (auto &image : images)
	{
		views.push_back(image.get().GetView());
		width  = std::max(width, image.get().GetWidth());
		height = std::max(height, image.get().GetHeight());
	}

	VkFramebufferCreateInfo frame_buffer_create_info = {};
	frame_buffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	frame_buffer_create_info.renderPass              = render_pass;
	frame_buffer_create_info.attachmentCount         = static_cast<uint32_t>(views.size());
	frame_buffer_create_info.pAttachments            = views.data();
	frame_buffer_create_info.width                   = width;
	frame_buffer_create_info.height                  = height;
	frame_buffer_create_info.layers                  = layers;

	vkCreateFramebuffer(m_device, &frame_buffer_create_info, nullptr, &m_handle);
}

Framebuffer::~Framebuffer()
{
	if (m_handle)
	{
		vkDestroyFramebuffer(m_device, m_handle, nullptr);
	}
}

Framebuffer::operator const VkFramebuffer &() const
{
	return m_handle;
}

const VkFramebuffer &Framebuffer::GetHandle() const
{
	return m_handle;
}
}        // namespace Ilum::Graphics