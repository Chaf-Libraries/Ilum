#include "FrameBuffer.hpp"

namespace Ilum
{
FrameBuffer &FrameBuffer::Bind(Texture *render_target, const TextureViewDesc &view_desc, const VkClearColorValue &clear)
{
	return *this;
}

void FrameBuffer::Bind(Texture *depth_stencil, const TextureViewDesc &view_desc, const VkClearDepthStencilValue &clear)
{
}

size_t FrameBuffer::Hash()
{
	if (m_dirty)
	{
		HashCombine(m_hash, m_width);
		HashCombine(m_hash, m_height);
		HashCombine(m_hash, m_layer);
		for (auto& view : m_views)
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
	}

	return m_hash;
}

}        // namespace Ilum