#include "RenderTarget.hpp"

namespace Ilum
{
Attachment::Attachment(uint32_t binding, const std::string &name, VkFormat format, bool multisampled) :
    m_binding(binding),
    m_name(name),
    m_format(format),
    m_multisampled(multisampled)
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

bool Attachment::isMultisampled() const
{
	return m_multisampled;
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

Subpass::Subpass(uint32_t binding, const std::vector<uint32_t> &attachment_bindings):
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
}        // namespace Ilum