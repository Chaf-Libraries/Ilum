#include "RenderTarget.hpp"
#include "Definitions.hpp"
#include "Device.hpp"
#include "Texture.hpp"

namespace Ilum::Vulkan
{
static std::unordered_map<size_t, VkRenderPass>  RenderPassCache;
static std::unordered_map<size_t, VkFramebuffer> FramebufferCache;

static uint32_t RenderTargetCount = 0;

RenderTarget::RenderTarget(RHIDevice *device) :
    RHIRenderTarget(device)
{
	RenderTargetCount++;
}

RenderTarget::~RenderTarget()
{
	if (--RenderTargetCount == 0)
	{
		p_device->WaitIdle();

		for (auto &[hash, render_pass] : RenderPassCache)
		{
			vkDestroyRenderPass(static_cast<Device *>(p_device)->GetDevice(), render_pass, nullptr);
		}

		for (auto &[hash, framebuffer] : FramebufferCache)
		{
			vkDestroyFramebuffer(static_cast<Device *>(p_device)->GetDevice(), framebuffer, nullptr);
		}

		RenderPassCache.clear();
		FramebufferCache.clear();
	}
}

RHIRenderTarget &RenderTarget::Set(uint32_t slot, RHITexture *texture, RHITextureDimension dimension, const ColorAttachment &attachment)
{
	return Set(slot, texture, TextureRange{dimension, 0, texture->GetDesc().mips, 0, texture->GetDesc().layers}, attachment);
}

RHIRenderTarget &RenderTarget::Set(uint32_t slot, RHITexture *texture, const TextureRange &range, const ColorAttachment &attachment)
{
	VkRenderingAttachmentInfo attachment_info = {};
	attachment_info.sType                     = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	attachment_info.loadOp                    = ToVulkanLoadOp[attachment.load];
	attachment_info.storeOp                   = ToVulkanStoreOp[attachment.store];
	attachment_info.imageView                 = static_cast<Texture *>(texture)->GetView(range);
	attachment_info.imageLayout               = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	std::memcpy(attachment_info.clearValue.color.float32, attachment.clear_value.data(), 4 * sizeof(float));

	HashCombine(m_hash, attachment_info.loadOp, attachment_info.storeOp, attachment_info.imageView, attachment_info.imageLayout);

	while (slot >= m_color_attachments.size())
	{
		m_color_attachments.push_back({});
		m_color_formats.push_back({});
	}

	m_color_attachments[slot] = attachment_info;
	m_color_formats[slot]     = ToVulkanFormat[texture->GetDesc().format];

	m_width  = std::max(m_width, texture->GetDesc().width);
	m_height = std::max(m_height, texture->GetDesc().height);
	m_layers = std::max(m_layers, texture->GetDesc().layers);

	return *this;
}

RHIRenderTarget &RenderTarget::Set(RHITexture *texture, RHITextureDimension dimension, const DepthStencilAttachment &attachment)
{
	return Set(texture, TextureRange{dimension, 0, texture->GetDesc().mips, 0, texture->GetDesc().layers}, attachment);
}

RHIRenderTarget &RenderTarget::Set(RHITexture *texture, const TextureRange &range, const DepthStencilAttachment &attachment)
{
	VkRenderingAttachmentInfo attachment_info     = {};
	attachment_info.sType                         = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	attachment_info.loadOp                        = ToVulkanLoadOp[attachment.depth_load];
	attachment_info.storeOp                       = ToVulkanStoreOp[attachment.depth_store];
	attachment_info.imageView                     = static_cast<Texture *>(texture)->GetView(range);
	attachment_info.imageLayout                   = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	attachment_info.clearValue.depthStencil.depth = attachment.clear_depth;

	HashCombine(m_hash, attachment_info.loadOp, attachment_info.storeOp, attachment_info.imageView, attachment_info.imageLayout);

	m_depth_attachment = attachment_info;
	m_depth_format     = ToVulkanFormat[texture->GetDesc().format];

	if (IsStencilFormat(texture->GetDesc().format))
	{
		attachment_info.loadOp                          = ToVulkanLoadOp[attachment.stencil_load];
		attachment_info.storeOp                         = ToVulkanStoreOp[attachment.stencil_store];
		attachment_info.clearValue.depthStencil.stencil = attachment.clear_stencil;
		m_stencil_attachment                            = attachment_info;
		m_stencil_format                                = ToVulkanFormat[texture->GetDesc().format];

		HashCombine(m_hash, attachment_info.loadOp, attachment_info.storeOp);
	}

	m_width  = std::max(m_width, texture->GetDesc().width);
	m_height = std::max(m_height, texture->GetDesc().height);
	m_layers = std::max(m_layers, texture->GetDesc().layers);

	return *this;
}

VkRenderPass RenderTarget::GetRenderPass() const
{
	std::vector<VkAttachmentDescription> descriptions;
	std::vector<VkAttachmentReference>   color_references;
	std::optional<VkAttachmentReference> depth_stencil_reference;

	for (uint32_t i = 0; i < m_color_attachments.size(); i++)
	{
		VkAttachmentDescription description = {};
		description.samples                 = VK_SAMPLE_COUNT_1_BIT;
		description.initialLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		description.finalLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		description.loadOp                  = m_color_attachments[i].loadOp;
		description.storeOp                 = m_color_attachments[i].storeOp;
		description.format                  = m_color_formats[i];

		VkAttachmentReference reference = {};
		reference.attachment            = static_cast<uint32_t>(descriptions.size());
		reference.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		descriptions.push_back(description);
		color_references.push_back(reference);
	}

	if (m_depth_attachment.has_value() || m_stencil_attachment.has_value())
	{
		VkAttachmentDescription description = {};
		description.samples                 = VK_SAMPLE_COUNT_1_BIT;
		description.initialLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		description.finalLayout             = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		description.format                  = m_depth_format.value();

		if (m_depth_attachment.has_value())
		{
			description.loadOp  = m_depth_attachment.value().loadOp;
			description.storeOp = m_depth_attachment.value().storeOp;
		}

		if (m_stencil_attachment.has_value())
		{
			description.stencilLoadOp  = m_stencil_attachment.value().loadOp;
			description.stencilStoreOp = m_stencil_attachment.value().storeOp;
		}

		VkAttachmentReference reference = {};
		reference.attachment            = static_cast<uint32_t>(descriptions.size());
		reference.layout                = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		descriptions.push_back(description);
		depth_stencil_reference = reference;
	}

	if (RenderPassCache.find(m_hash) != RenderPassCache.end())
	{
		return RenderPassCache.at(m_hash);
	}

	VkRenderPass render_pass = VK_NULL_HANDLE;

	VkSubpassDescription subpass_description    = {};
	subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass_description.colorAttachmentCount    = static_cast<uint32_t>(color_references.size());
	subpass_description.pColorAttachments       = color_references.data();
	subpass_description.pDepthStencilAttachment = depth_stencil_reference.has_value() ? &depth_stencil_reference.value() : nullptr;

	std::array<VkSubpassDependency, 2> subpass_dependencies;

	subpass_dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
	subpass_dependencies[0].dstSubpass      = 0;
	subpass_dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	subpass_dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	subpass_dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
	subpass_dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	subpass_dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	subpass_dependencies[1].srcSubpass      = 0;
	subpass_dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
	subpass_dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	subpass_dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	subpass_dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	subpass_dependencies[1].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
	subpass_dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	// Create render pass
	VkRenderPassCreateInfo render_pass_create_info = {};
	render_pass_create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_create_info.attachmentCount        = static_cast<uint32_t>(descriptions.size());
	render_pass_create_info.pAttachments           = descriptions.data();
	render_pass_create_info.subpassCount           = 1;
	render_pass_create_info.pSubpasses             = &subpass_description;
	render_pass_create_info.dependencyCount        = static_cast<uint32_t>(subpass_dependencies.size());
	render_pass_create_info.pDependencies          = subpass_dependencies.data();

	vkCreateRenderPass(static_cast<Device *>(p_device)->GetDevice(), &render_pass_create_info, nullptr, &render_pass);

	RenderPassCache.emplace(m_hash, render_pass);

	return render_pass;
}

VkFramebuffer RenderTarget::GetFramebuffer() const
{
	std::vector<VkImageView> views;
	views.reserve(m_color_attachments.size() + 1);
	for (auto &attachment : m_color_attachments)
	{
		views.push_back(attachment.imageView);
	}
	if (m_depth_attachment.has_value())
	{
		views.push_back(m_depth_attachment.value().imageView);
	}

	size_t hash = 0;
	HashCombine(hash, views);

	if (FramebufferCache.find(hash) != FramebufferCache.end())
	{
		return FramebufferCache.at(hash);
	}

	VkFramebufferCreateInfo frame_buffer_create_info = {};
	frame_buffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	frame_buffer_create_info.renderPass              = GetRenderPass();
	frame_buffer_create_info.attachmentCount         = static_cast<uint32_t>(views.size());
	frame_buffer_create_info.pAttachments            = views.data();
	frame_buffer_create_info.width                   = m_width;
	frame_buffer_create_info.height                  = m_height;
	frame_buffer_create_info.layers                  = m_layers;

	VkFramebuffer frame_buffer = VK_NULL_HANDLE;
	vkCreateFramebuffer(static_cast<Device *>(p_device)->GetDevice(), &frame_buffer_create_info, nullptr, &frame_buffer);

	FramebufferCache.emplace(hash, frame_buffer);

	return frame_buffer;
}

VkRect2D RenderTarget::GetRenderArea() const
{
	return VkRect2D{
	    VkOffset2D{0, 0},
	    VkExtent2D{m_width, m_height}};
}

std::vector<VkClearValue> RenderTarget::GetClearValue() const
{
	std::vector<VkClearValue> clear_values;
	clear_values.reserve(m_color_attachments.size() + 1);
	for (auto &attachment : m_color_attachments)
	{
		clear_values.push_back(attachment.clearValue);
	}
	if (m_depth_attachment.has_value())
	{
		clear_values.push_back(m_depth_attachment.value().clearValue);
	}

	return clear_values;
}

const std::vector<VkRenderingAttachmentInfo> &RenderTarget::GetColorAttachments()
{
	return m_color_attachments;
}

const std::optional<VkRenderingAttachmentInfo> &RenderTarget::GetDepthAttachment()
{
	return m_depth_attachment;
}

const std::optional<VkRenderingAttachmentInfo> &RenderTarget::GetStencilAttachment()
{
	return m_stencil_attachment;
}

const std::vector<VkFormat> &RenderTarget::GetColorFormats()
{
	return m_color_formats;
}

const std::optional<VkFormat> &RenderTarget::GetDepthFormat()
{
	return m_depth_format;
}

const std::optional<VkFormat> &RenderTarget::GetStencilFormat()
{
	return m_stencil_format;
}

size_t RenderTarget::GetFormatHash() const
{
	size_t hash = 0;
	HashCombine(hash, m_color_formats);

	if (m_depth_format.has_value())
	{
		HashCombine(hash, m_depth_format.value());
	}

	if (m_stencil_format.has_value())
	{
		HashCombine(hash, m_stencil_format.value());
	}

	return hash;
}

size_t RenderTarget::GetHash() const
{
	return m_hash;
}

RHIRenderTarget &RenderTarget::Clear()
{
	m_color_attachments.clear();
	m_depth_attachment.reset();
	m_stencil_attachment.reset();

	m_color_formats.clear();
	m_depth_format.reset();
	m_stencil_format.reset();

	m_width  = 0;
	m_height = 0;

	m_hash = 0;

	return *this;
}
}        // namespace Ilum::Vulkan