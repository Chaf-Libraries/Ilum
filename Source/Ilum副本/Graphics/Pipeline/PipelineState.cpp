#include "PipelineState.hpp"

namespace Ilum
{
PipelineState &PipelineState::addOutputAttachment(const std::string &name, VkClearColorValue clear, uint32_t layer)
{
	m_output_attachments.push_back(OutputAttachment{
	    name,
	    clear,
	    {},
	    AttachmentState::Clear_Color,
	    layer});
	return *this;
}

PipelineState &PipelineState::addOutputAttachment(const std::string &name, VkClearDepthStencilValue clear, uint32_t layer)
{
	m_output_attachments.push_back(OutputAttachment{
	    name,
	    {},
	    clear,
	    AttachmentState::Clear_Depth_Stencil,
	    layer});
	return *this;
}

PipelineState &PipelineState::addOutputAttachment(const std::string &name, AttachmentState state, uint32_t layer)
{
	m_output_attachments.push_back(OutputAttachment{
	    name,
	    {},
	    {},
	    state,
	    layer});
	return *this;
}

PipelineState &PipelineState::addDependency(const std::string &name, VkBufferUsageFlagBits usage)
{
	m_buffer_dependencies.push_back(BufferDependency{name, usage});
	return *this;
}

PipelineState &PipelineState::addDependency(const std::string &name, VkImageUsageFlagBits usage)
{
	m_image_dependencies.push_back(ImageDependency{name, usage});
	return *this;
}

PipelineState &PipelineState::declareAttachment(const std::string &name, VkFormat format, uint32_t width, uint32_t height, bool mipmaps, uint32_t layers)
{
	m_attachment_declarations.push_back(AttachmentDeclaration{
	    name,
	    format,
	    width,
	    height,
	    mipmaps,
	    layers});
	return *this;
}

const std::vector<PipelineState::ImageDependency> &PipelineState::getImageDependencies() const
{
	return m_image_dependencies;
}

const std::vector<PipelineState::BufferDependency> &PipelineState::getBufferDependencies() const
{
	return m_buffer_dependencies;
}

const std::vector<PipelineState::AttachmentDeclaration> &PipelineState::getAttachmentDeclarations() const
{
	return m_attachment_declarations;
}

const std::vector<PipelineState::OutputAttachment> &PipelineState::getOutputAttachments() const
{
	return m_output_attachments;
}

}        // namespace Ilum