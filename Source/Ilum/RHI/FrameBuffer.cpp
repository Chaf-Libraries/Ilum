#include "FrameBuffer.hpp"

namespace Ilum
{
uint32_t FrameBuffer::GetWidth() const
{
	return m_width;
}

uint32_t FrameBuffer::GetHeight() const
{
	return m_height;
}

uint32_t FrameBuffer::GetLayer() const
{
	return m_layer;
}

const std::vector<VkClearValue> &FrameBuffer::GetClearValue() const
{
	return m_clear_values;
}

FrameBuffer &FrameBuffer::Bind(Texture *render_target, const TextureViewDesc &view_desc, const ColorAttachmentInfo &info)
{
	VkClearValue clear_value = {};
	clear_value.color        = info.clear_value;

	m_width  = std::max(m_width, render_target->GetWidth());
	m_height = std::max(m_height, render_target->GetHeight());
	m_layer  = std::max(m_layer, view_desc.layer_count);

	m_views.push_back(render_target->GetView(view_desc));
	m_clear_values.push_back(clear_value);

	VkAttachmentDescription description = {};
	description.samples                 = info.samples;
	description.format                  = render_target->GetFormat();
	description.loadOp                  = info.load_op;
	description.storeOp                 = info.store_op;
	description.stencilLoadOp           = info.stencil_load_op;
	description.stencilStoreOp          = info.stencil_store_op;
	description.initialLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	description.finalLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	m_attachment_descriptions.push_back(description);

	VkAttachmentReference reference = {};
	reference.attachment            = static_cast<uint32_t>(m_views.size()) - 1;
	reference.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	m_attachment_references.push_back(reference);

	m_dirty = true;

	return *this;
}

FrameBuffer &FrameBuffer::Bind(Texture *depth_stencil, const TextureViewDesc &view_desc, const DepthStencilAttachmentInfo &info)
{
	VkClearValue clear_value = {};
	clear_value.depthStencil = info.clear_value;

	m_width  = std::max(m_width, depth_stencil->GetWidth());
	m_height = std::max(m_height, depth_stencil->GetHeight());
	m_layer  = std::max(m_layer, depth_stencil->GetLayerCount());

	m_views.push_back(depth_stencil->GetView(view_desc));
	m_clear_values.push_back(clear_value);

	VkAttachmentDescription description = {};
	description.samples                 = info.samples;
	description.format                  = depth_stencil->GetFormat();
	description.loadOp                  = info.load_op;
	description.storeOp                 = info.store_op;
	description.stencilLoadOp           = info.stencil_load_op;
	description.stencilStoreOp          = info.stencil_store_op;
	description.initialLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	description.finalLayout             = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	m_attachment_descriptions.push_back(description);

	VkAttachmentReference reference = {};
	reference.attachment            = static_cast<uint32_t>(m_views.size()) - 1;
	reference.layout                     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	m_depth_stencil_attachment_reference = reference;

	m_dirty = true;

	return *this;
}

size_t FrameBuffer::Hash()
{
	if (!m_dirty)
	{
		return m_hash;
	}

	HashCombine(m_hash, m_width);
	HashCombine(m_hash, m_height);
	HashCombine(m_hash, m_layer);
	for (auto &view : m_views)
	{
		HashCombine(m_hash, view);
	}
	for (auto &attachment_description : m_attachment_descriptions)
	{
		HashCombine(m_hash, attachment_description.format);
		HashCombine(m_hash, attachment_description.samples);
		HashCombine(m_hash, attachment_description.loadOp);
		HashCombine(m_hash, attachment_description.storeOp);
		HashCombine(m_hash, attachment_description.stencilLoadOp);
		HashCombine(m_hash, attachment_description.stencilStoreOp);
	}
	for (auto &attachment_reference : m_attachment_references)
	{
		HashCombine(m_hash, attachment_reference.attachment);
		HashCombine(m_hash, attachment_reference.layout);
	}
	if (m_depth_stencil_attachment_reference.has_value())
	{
		HashCombine(m_hash, m_depth_stencil_attachment_reference.value().attachment);
		HashCombine(m_hash, m_depth_stencil_attachment_reference.value().layout);
	}

	m_dirty = false;

	return m_hash;
}

}        // namespace Ilum