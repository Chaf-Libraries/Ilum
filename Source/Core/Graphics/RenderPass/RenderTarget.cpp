#include "RenderTarget.hpp"
#include "RenderPass.hpp"

#include "Core/Graphics/Image/Image2D.hpp"
#include "Core/Graphics/Image/ImageDepth.hpp"

namespace Ilum
{
Attachment::Attachment(uint32_t binding, const std::string &name, Type type, VkFormat format, const Math::Rgba &clear_color, VkSampleCountFlagBits samples) :
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

const Math::Rgba &Attachment::getColor() const
{
	return m_clear_color;
}

RenderArea::RenderArea(const Math::Vector2 &extent, const Math::Vector2 &offset) :
    m_extent(extent), m_offset(offset)
{
}

bool RenderArea::operator==(const RenderArea &rhs) const
{
	return m_extent == rhs.m_extent && m_offset == rhs.m_offset;
}

bool RenderArea::operator!=(const RenderArea &rhs) const
{
	return !(*this == rhs);
}

const Math::Vector2 &RenderArea::getExtent() const
{
	return m_extent;
}

void RenderArea::setExtent(const Math::Vector2 &extent)
{
	m_extent = extent;
}

const Math::Vector2 &RenderArea::getOffset() const
{
	return m_offset;
}

void RenderArea::setOffset(const Math::Vector2 &offset)
{
	m_offset = offset;
}

Subpass::Subpass(uint32_t binding, const std::vector<uint32_t> &attachment_bindings) :
    m_binding(binding), m_attachment_bindings(attachment_bindings)
{
}

uint32_t Subpass::getBinding() const
{
	return m_binding;
}

const std::vector<uint32_t> &Subpass::getAttachmentBindings() const
{
	return m_attachment_bindings;
}

RenderTarget::RenderTarget(const std::vector<Attachment> &attachments, const std::vector<Subpass> &subpasses, const RenderArea &render_area) :
    m_attachments(attachments),
    m_subpasses(subpasses),
    m_render_area(render_area)
{
}

const RenderArea &RenderTarget::getRenderArea() const
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
	if (m_color_attachments.size() >= idx)
	{
		return nullptr;
	}

	return m_color_attachments[idx].get();
}
}        // namespace Ilum