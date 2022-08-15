#include "RenderTarget.hpp"
#include "Definitions.hpp"
#include "Device.hpp"
#include "Texture.hpp"

namespace Ilum::Vulkan
{
static std::unordered_map<size_t, VkRenderPass>  RenderPassCache;
static std::unordered_map<size_t, VkFramebuffer> FramebufferCache;

RenderTarget::RenderTarget(RHIDevice *device) :
    RHIRenderTarget(device)
{
}

RHIRenderTarget &RenderTarget::Add(RHITexture *texture, RHITextureDimension dimension, const ColorAttachment &attachment)
{
	return Add(texture, TextureRange{dimension, 0, texture->GetDesc().mips, 0, texture->GetDesc().layers}, attachment);
}

RHIRenderTarget &RenderTarget::Add(RHITexture *texture, const TextureRange &range, const ColorAttachment &attachment)
{
	VkAttachmentDescription description = {};
	description.format                  = ToVulkanFormat[texture->GetDesc().format];
	description.initialLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	description.finalLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	description.loadOp                  = ToVulkanLoadOp[attachment.load];
	description.storeOp                 = ToVulkanStoreOp[attachment.store];
	description.stencilLoadOp           = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	description.stencilStoreOp          = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	description.samples                 = VK_SAMPLE_COUNT_1_BIT;

	VkAttachmentReference reference = {};
	reference.attachment            = static_cast<uint32_t>(m_descriptions.size());
	reference.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	FrameBufferResolve framebuffer_resolve = {};
	framebuffer_resolve.view               = static_cast<Texture *>(texture)->GetView(range);
	for (uint32_t i = 0; i < 4; i++)
	{
		framebuffer_resolve.clear_value.color.float32[i] = attachment.clear_value[i];
	}

	HashCombine(
	    m_render_pass_hash,
	    description.format,
	    description.initialLayout,
	    description.finalLayout,
	    description.loadOp,
	    description.storeOp,
	    description.samples,
	    reference.attachment,
	    reference.layout);

	HashCombine(m_framebuffer_hash, framebuffer_resolve.view);

	m_descriptions.push_back(description);
	m_color_reference.push_back(reference);
	m_framebuffer_resolves.push_back(framebuffer_resolve);

	m_width  = std::max(m_width, texture->GetDesc().width);
	m_height = std::max(m_height, texture->GetDesc().height);
	m_layers = std::max(m_layers, texture->GetDesc().layers);

	return *this;
}

RHIRenderTarget &RenderTarget::Add(RHITexture *texture, RHITextureDimension dimension, const DepthStencilAttachment &attachment)
{
	return Add(texture, TextureRange{dimension, 0, texture->GetDesc().mips, 0, texture->GetDesc().layers}, attachment);
}

RHIRenderTarget &RenderTarget::Add(RHITexture *texture, const TextureRange &range, const DepthStencilAttachment &attachment)
{
	VkAttachmentDescription description = {};
	description.format                  = ToVulkanFormat[texture->GetDesc().format];
	description.initialLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	description.finalLayout             = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	description.loadOp                  = ToVulkanLoadOp[attachment.depth_load];
	description.storeOp                 = ToVulkanStoreOp[attachment.depth_store];
	description.samples                 = VK_SAMPLE_COUNT_1_BIT;

	VkAttachmentReference reference = {};
	reference.attachment            = static_cast<uint32_t>(m_descriptions.size());
	reference.layout                = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	if (IsStencilFormat(texture->GetDesc().format))
	{
		description.stencilLoadOp = ToVulkanLoadOp[attachment.stencil_load];
		description.stencilStoreOp = ToVulkanStoreOp[attachment.stencil_store];
	}

	reference.attachment = static_cast<uint32_t>(m_descriptions.size());
	reference.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	FrameBufferResolve framebuffer_resolve               = {};
	framebuffer_resolve.view                             = static_cast<Texture *>(texture)->GetView(range);
	framebuffer_resolve.clear_value.depthStencil.depth   = attachment.clear_depth;
	framebuffer_resolve.clear_value.depthStencil.stencil = attachment.clear_stencil;

	HashCombine(
	    m_render_pass_hash,
	    description.format,
	    description.initialLayout,
	    description.finalLayout,
	    description.loadOp,
	    description.storeOp,
	    description.stencilLoadOp,
	    description.stencilStoreOp,
	    description.samples,
	    reference.attachment,
	    reference.layout);

	HashCombine(m_framebuffer_hash, framebuffer_resolve.view);

	m_depth_stencil_reference = reference;
	m_descriptions.push_back(description);
	m_framebuffer_resolves.push_back(framebuffer_resolve);

	m_width  = std::max(m_width, texture->GetDesc().width);
	m_height = std::max(m_height, texture->GetDesc().height);
	m_layers = std::max(m_layers, texture->GetDesc().layers);

	return *this;
}

VkRenderPass RenderTarget::GetRenderPass() const
{
	if (RenderPassCache.find(m_render_pass_hash) != RenderPassCache.end())
	{
		return RenderPassCache.at(m_render_pass_hash);
	}

	VkRenderPass render_pass = VK_NULL_HANDLE;

	VkSubpassDescription subpass_description    = {};
	subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass_description.colorAttachmentCount    = static_cast<uint32_t>(m_color_reference.size());
	subpass_description.pColorAttachments       = m_color_reference.data();
	subpass_description.pDepthStencilAttachment = m_depth_stencil_reference.has_value() ? &m_depth_stencil_reference.value() : nullptr;

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
	render_pass_create_info.attachmentCount        = static_cast<uint32_t>(m_descriptions.size());
	render_pass_create_info.pAttachments           = m_descriptions.data();
	render_pass_create_info.subpassCount           = 1;
	render_pass_create_info.pSubpasses             = &subpass_description;
	render_pass_create_info.dependencyCount        = static_cast<uint32_t>(subpass_dependencies.size());
	render_pass_create_info.pDependencies          = subpass_dependencies.data();

	vkCreateRenderPass(static_cast<Device *>(p_device)->GetDevice(), &render_pass_create_info, nullptr, &render_pass);

	RenderPassCache.emplace(m_render_pass_hash, render_pass);

	return render_pass;
}

VkFramebuffer RenderTarget::GetFramebuffer() const
{
	if (FramebufferCache.find(m_framebuffer_hash) != FramebufferCache.end())
	{
		return FramebufferCache.at(m_framebuffer_hash);
	}

	std::vector<VkImageView> views;
	views.reserve(m_framebuffer_resolves.size());
	for (auto &resolve : m_framebuffer_resolves)
	{
		views.push_back(resolve.view);
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

	FramebufferCache.emplace(m_framebuffer_hash, frame_buffer);

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
	clear_values.reserve(m_framebuffer_resolves.size());
	for (auto &resolve : m_framebuffer_resolves)
	{
		clear_values.push_back(resolve.clear_value);
	}

	return clear_values;
}
}        // namespace Ilum::Vulkan