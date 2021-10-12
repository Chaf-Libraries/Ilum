#include "RenderTarget.hpp"
#include "Framebuffer.hpp"
#include "RenderPass.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Image/Image2D.hpp"
#include "Graphics/Image/ImageDepth.hpp"
#include "Graphics/RenderPass/Swapchain.hpp"

namespace Ilum
{
Attachment::Attachment(uint32_t binding, const std::string &name, Type type, VkFormat format, const Rgba &clear_color, VkSampleCountFlagBits samples) :
    m_binding(binding),
    m_name(name),
    m_type(type),
    m_format(format),
    m_clear_color(clear_color),
    m_samples(samples)
{
}

uint32_t Attachment::getBinding() const
{
	return m_binding;
}

const std::string &Attachment::getName() const
{
	return m_name;
}

Attachment::Type Attachment::getType() const
{
	return m_type;
}

VkSampleCountFlagBits Attachment::getSamples() const
{
	return m_samples;
}

VkFormat Attachment::getFormat() const
{
	return m_format;
}

const Rgba &Attachment::getColor() const
{
	return m_clear_color;
}

Subpass::Subpass(uint32_t index, std::vector<uint32_t> &&output_attachments, std::vector<uint32_t> &&input_attachments) :
    m_index(index), m_output_attachments(std::move(output_attachments)), m_input_attachments(std::move(input_attachments))
{
}

uint32_t Subpass::getIndex() const
{
	return m_index;
}

const std::vector<uint32_t> &Subpass::getInputAttachments() const
{
	return m_input_attachments;
}

const std::vector<uint32_t> &Subpass::getOutputAttachments() const
{
	return m_output_attachments;
}

RenderTarget::RenderTarget(std::vector<Attachment> &&attachments, std::vector<Subpass> &&subpasses, const VkRect2D &render_area) :
    m_attachments(std::move(attachments)),
    m_subpasses(std::move(subpasses)),
    m_render_area(render_area),
    m_subpass_attachment_counts(m_subpasses.size())
{
	for (const auto &attachment : m_attachments)
	{
		VkClearValue clear_value = {};
		switch (attachment.getType())
		{
			case Attachment::Type::Image:
				clear_value.color = {{attachment.getColor().x, attachment.getColor().y, attachment.getColor().z, attachment.getColor().w}};
				break;
			case Attachment::Type::Depth:
				clear_value.depthStencil = {1.0f, 0};
				m_depth_attachment       = attachment;
				break;
			case Attachment::Type::Swapchain:
				clear_value.color      = {{attachment.getColor().x, attachment.getColor().y, attachment.getColor().z, attachment.getColor().w}};
				m_swapchain_attachment = attachment;
				break;
			default:
				break;
		}

		m_clear_values.push_back(clear_value);

		if (attachment.getType() == Attachment::Type::Depth)
		{
			continue;
		}

		for (const auto &subpass : m_subpasses)
		{
			auto &subpass_bindings = subpass.getOutputAttachments();
			if (std::find(subpass_bindings.begin(), subpass_bindings.end(), attachment.getBinding()) != subpass_bindings.end())
			{
				m_subpass_attachment_counts[subpass.getIndex()]++;
			}
		}
	}

	build();
}

void RenderTarget::resize(const VkRect2D &render_area)
{
	if (render_area.extent.width != m_render_area.extent.width ||
	    render_area.extent.height != m_render_area.extent.height ||
	    render_area.offset.x != m_render_area.offset.x ||
	    render_area.offset.y != m_render_area.offset.y)
	{
		m_render_area = render_area;
		build();
	}
}

const VkRect2D &RenderTarget::getRenderArea() const
{
	return m_render_area;
}

const std::vector<Attachment> &RenderTarget::getAttachments() const
{
	return m_attachments;
}

std::optional<Attachment> RenderTarget::getAttachment(uint32_t binding) const
{
	auto it = std::find_if(m_attachments.begin(), m_attachments.end(), [binding](const Attachment &attachment) { return attachment.getBinding() == binding; });

	if (it != m_attachments.end())
	{
		return *it;
	}

	return std::nullopt;
}

std::optional<Attachment> RenderTarget::getAttachment(const std::string &name) const
{
	auto it = std::find_if(m_attachments.begin(), m_attachments.end(), [name](const Attachment &attachment) { return attachment.getName() == name; });

	if (it != m_attachments.end())
	{
		return *it;
	}

	return std::nullopt;
}

const std::vector<Subpass> &RenderTarget::getSubpasses() const
{
	return m_subpasses;
}

const ImageDepth *RenderTarget::getDepthStencil() const
{
	return m_depth_stencil.get();
}

const Image2D *RenderTarget::getColorAttachment(uint32_t idx) const
{
	if (m_color_attachments_mapping.find(idx) == m_color_attachments_mapping.end())
	{
		return nullptr;
	}

	return m_color_attachments_mapping.at(idx);
}

const Image2D *RenderTarget::getColorAttachment(const std::string &name) const
{
	for (auto& attachment : m_attachments)
	{
		if (attachment.getName() == name && attachment.getType() == Attachment::Type::Image)
		{
			return m_color_attachments.at(attachment.getBinding()).get();
		}
	}

	return nullptr;
}

const VkRenderPass &RenderTarget::getRenderPass() const
{
	return *m_render_pass;
}

const std::vector<uint32_t> &RenderTarget::getSubpassAttachmentCounts() const
{
	return m_subpass_attachment_counts;
}

const std::vector<VkClearValue> &RenderTarget::getClearValue() const
{
	return m_clear_values;
}

const VkFramebuffer &RenderTarget::getCurrentFramebuffer() const
{
	return m_framebuffer->getFramebuffers()[GraphicsContext::instance()->getSwapchain().getActiveImageIndex()];
}

bool RenderTarget::hasSwapchainAttachment() const
{
	return m_swapchain_attachment.has_value();
}

bool RenderTarget::hasDepthAttachment() const
{
	return m_depth_attachment.has_value();
}

void RenderTarget::build()
{
	if (m_depth_attachment)
	{
		m_depth_stencil = createScope<ImageDepth>(m_render_area.extent.width, m_render_area.extent.height, m_depth_attachment->getSamples());
	}

	m_color_attachments.clear();
	m_color_attachments_mapping.clear();

	for (auto &attachment : m_attachments)
	{
		if (attachment.getType() == Attachment::Type::Image)
		{
			// Is input attachment?
			VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
			for (auto &subpass : m_subpasses)
			{
				if (std::find(subpass.getInputAttachments().begin(), subpass.getInputAttachments().end(), attachment.getBinding()) != subpass.getInputAttachments().end())
				{
					usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
				}
			}

			m_color_attachments.emplace_back(createScope<Image2D>(m_render_area.extent.width, m_render_area.extent.height, attachment.getFormat(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, usage));
			m_color_attachments_mapping.insert({attachment.getBinding(), m_color_attachments.back().get()});
		}
	}

	if (!m_render_pass)
	{
		m_render_pass = createScope<RenderPass>(*this);
	}

	m_framebuffer = createScope<Framebuffer>(*this, *m_render_pass);
}
}        // namespace Ilum