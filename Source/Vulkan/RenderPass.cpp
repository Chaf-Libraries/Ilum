#include "RenderPass.hpp"
#include "Device.hpp"
#include "RenderContext.hpp"

#include <Core/Hash.hpp>

#include <optional>

namespace std
{
template <>
struct hash<Ilum::Vulkan::Attachment>
{
	size_t operator()(const Ilum::Vulkan::Attachment &attachment) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, static_cast<size_t>(attachment.format));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(attachment.samples));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(attachment.load_store.loadOp));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(attachment.load_store.storeOp));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(attachment.stencil_load_store.loadOp));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(attachment.stencil_load_store.storeOp));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(attachment.initialLayout));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(attachment.finalLayout));
		return seed;
	}
};

template <>
struct hash<Ilum::Vulkan::SubpassInfo>
{
	size_t operator()(const Ilum::Vulkan::SubpassInfo &subpass_info) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, subpass_info.subpass_name);
		for (auto &attachment : subpass_info.input_attachments)
		{
			Ilum::Core::HashCombine(seed, attachment);
		}
		for (auto &attachment : subpass_info.output_attachments)
		{
			Ilum::Core::HashCombine(seed, attachment);
		}
		return seed;
	}
};
}        // namespace std

namespace Ilum::Vulkan
{
RenderPass::RenderPass(const std::vector<Attachment> &attachments, const std::vector<SubpassInfo> &subpass_infos)
{
	std::vector<VkAttachmentDescription> attachment_descriptions;
	std::vector<VkAttachmentReference>   attachment_references;

	for (uint32_t i = 0; i < attachments.size(); i++)
	{
		VkAttachmentDescription attachment_description = {};
		attachment_description.format                  = attachments[i].format;
		attachment_description.samples                 = attachments[i].samples;
		attachment_description.loadOp                  = attachments[i].load_store.loadOp;
		attachment_description.storeOp                 = attachments[i].load_store.storeOp;
		attachment_description.stencilLoadOp           = attachments[i].stencil_load_store.loadOp;
		attachment_description.stencilStoreOp          = attachments[i].stencil_load_store.storeOp;
		attachment_description.initialLayout           = attachments[i].initialLayout;
		attachment_description.finalLayout             = attachments[i].finalLayout;
		attachment_descriptions.push_back(attachment_description);

		VkAttachmentReference attachment_reference = {};
		attachment_reference.attachment            = i;
		attachment_reference.layout                = attachments[i].finalLayout;

		attachment_references.push_back(attachment_reference);
	}

	std::vector<VkSubpassDescription> subpass_descriptions;
	for (uint32_t i = 0; i < subpass_infos.size(); i++)
	{
		std::vector<VkAttachmentReference>   input_attachments;
		std::vector<VkAttachmentReference>   color_ouput_attachments;
		std::vector<VkAttachmentReference>   color_resolve_attachments;
		std::optional<VkAttachmentReference> depth_stencil_output_attachment;

		VkSubpassDescription subpass_description = {};
		subpass_description.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
		// Resolve input attachment
		for (uint32_t attachment_index : subpass_infos[i].input_attachments)
		{
			input_attachments.push_back(attachment_references[attachment_index]);
		}
		// Resolve output attachment
		for (uint32_t attachment_index : subpass_infos[i].output_attachments)
		{
			if (attachment_references[attachment_index].layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
			{
				// output - depth_stencil
				depth_stencil_output_attachment = attachment_references[attachment_index];
			}
			else
			{
				// output - color
				color_ouput_attachments.push_back(attachment_references[attachment_index]);
			}
		}

		subpass_description.inputAttachmentCount    = static_cast<uint32_t>(input_attachments.size());
		subpass_description.pInputAttachments       = input_attachments.data();
		subpass_description.colorAttachmentCount    = static_cast<uint32_t>(input_attachments.size());
		subpass_description.pColorAttachments       = input_attachments.data();
		subpass_description.pDepthStencilAttachment = depth_stencil_output_attachment.has_value() ? &depth_stencil_output_attachment.value() : nullptr;

		subpass_descriptions.push_back(subpass_description);
	}

	std::vector<VkSubpassDependency> dependencies(subpass_infos.size() + 1);

	dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass      = 0;
	dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	for (uint32_t i = 1; i < dependencies.size(); i++)
	{
		dependencies[i].srcSubpass      = i - 1;
		dependencies[i].dstSubpass      = i;
		dependencies[i].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[i].dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[i].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[i].dstAccessMask   = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
		dependencies[i].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	}

	dependencies[dependencies.size()].srcSubpass      = dependencies.size();
	dependencies[dependencies.size()].dstSubpass      = VK_SUBPASS_EXTERNAL;
	dependencies[dependencies.size()].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[dependencies.size()].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[dependencies.size()].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[dependencies.size()].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[dependencies.size()].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	VkRenderPassCreateInfo render_pass_create_info = {};
	render_pass_create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_create_info.attachmentCount        = static_cast<uint32_t>(attachment_descriptions.size());
	render_pass_create_info.pAttachments           = attachment_descriptions.data();
	render_pass_create_info.subpassCount           = static_cast<uint32_t>(subpass_descriptions.size());
	render_pass_create_info.pSubpasses             = subpass_descriptions.data();
	render_pass_create_info.dependencyCount        = static_cast<uint32_t>(dependencies.size());
	render_pass_create_info.pDependencies          = dependencies.data();

	vkCreateRenderPass(RenderContext::GetDevice(), &render_pass_create_info, nullptr, &m_handle);

	// Update hash
	m_hash = 0;
	for (auto &attachment : attachments)
	{
		Core::HashCombine(m_hash, attachment);
	}
	for (auto &subpass_info : subpass_infos)
	{
		Core::HashCombine(m_hash, subpass_info);
	}
}

RenderPass::RenderPass(const std::vector<Attachment> &attachments)
{
	std::vector<VkAttachmentDescription> attachment_descriptions;
	std::vector<VkAttachmentReference>   color_attachment_references;
	std::optional<VkAttachmentReference> depth_stencil_attachment_reference;

	for (uint32_t i = 0; i < attachments.size(); i++)
	{
		VkAttachmentDescription attachment_description = {};
		attachment_description.format                  = attachments[i].format;
		attachment_description.samples                 = attachments[i].samples;
		attachment_description.loadOp                  = attachments[i].load_store.loadOp;
		attachment_description.storeOp                 = attachments[i].load_store.storeOp;
		attachment_description.stencilLoadOp           = attachments[i].stencil_load_store.loadOp;
		attachment_description.stencilStoreOp          = attachments[i].stencil_load_store.storeOp;
		attachment_description.initialLayout           = attachments[i].initialLayout;
		attachment_description.finalLayout             = attachments[i].finalLayout;
		attachment_descriptions.push_back(attachment_description);

		VkAttachmentReference attachment_reference = {};
		attachment_reference.attachment            = i;
		attachment_reference.layout                = attachments[i].finalLayout;

		if (attachment_reference.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			depth_stencil_attachment_reference = attachment_reference;
		}
		else
		{
			color_attachment_references.push_back(attachment_reference);
		}
	}

	VkSubpassDescription subpass_description    = {};
	subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass_description.colorAttachmentCount    = static_cast<uint32_t>(color_attachment_references.size());
	subpass_description.pColorAttachments       = color_attachment_references.data();
	subpass_description.pDepthStencilAttachment = depth_stencil_attachment_reference.has_value() ? &depth_stencil_attachment_reference.value() : nullptr;

	VkSubpassDependency dependencies[2];
	dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass      = 0;
	dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass      = 0;
	dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	VkRenderPassCreateInfo render_pass_create_info = {};
	render_pass_create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_create_info.attachmentCount        = static_cast<uint32_t>(attachment_descriptions.size());
	render_pass_create_info.pAttachments           = attachment_descriptions.data();
	render_pass_create_info.subpassCount           = 1;
	render_pass_create_info.pSubpasses             = &subpass_description;
	render_pass_create_info.dependencyCount        = 2;
	render_pass_create_info.pDependencies          = dependencies;

	vkCreateRenderPass(RenderContext::GetDevice(), &render_pass_create_info, nullptr, &m_handle);

	// Update hash
	m_hash = 0;
	for (auto &attachment : attachments)
	{
		Core::HashCombine(m_hash, attachment);
	}
}

RenderPass::~RenderPass()
{
	if (m_handle)
	{
		vkDestroyRenderPass(RenderContext::GetDevice(), m_handle, nullptr);
	}
}

RenderPass::operator const VkRenderPass &() const
{
	return m_handle;
}

const VkRenderPass &RenderPass::GetHandle() const
{
	return m_handle;
}

size_t RenderPass::GetHash() const
{
	return m_hash;
}

Framebuffer::Framebuffer(const std::vector<VkImageView> &views, const RenderPass &render_pass, uint32_t width, uint32_t height, uint32_t layers)
{
	VkFramebufferCreateInfo frame_buffer_create_info = {};
	frame_buffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	frame_buffer_create_info.renderPass              = render_pass;
	frame_buffer_create_info.attachmentCount         = static_cast<uint32_t>(views.size());
	frame_buffer_create_info.pAttachments            = views.data();
	frame_buffer_create_info.width                   = width;
	frame_buffer_create_info.height                  = height;
	frame_buffer_create_info.layers                  = layers;

	vkCreateFramebuffer(RenderContext::GetDevice(), &frame_buffer_create_info, nullptr, &m_handle);
}

Framebuffer::~Framebuffer()
{
	if (m_handle)
	{
		vkDestroyFramebuffer(RenderContext::GetDevice(), m_handle, nullptr);
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
}        // namespace Ilum::Vulkan