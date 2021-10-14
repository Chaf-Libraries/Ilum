//#include "RenderPass.hpp"
//#include "RenderTarget.hpp"
//
//#include "Device/LogicalDevice.hpp"
//#include "Device/Surface.hpp"
//
//#include "Graphics/GraphicsContext.hpp"
//#include "Graphics/Image/ImageDepth.hpp"
//
//namespace Ilum
//{
//RenderPass::RenderPass(const RenderTarget &render_target)
//{
//	std::vector<VkAttachmentDescription> attachment_descriptions;
//
//	// Create render pass attachment description
//	for (const auto &attachment : render_target.getAttachments())
//	{
//		VkAttachmentDescription attachment_description = {};
//		attachment_description.samples                 = attachment.getSamples();
//		attachment_description.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
//		attachment_description.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
//		attachment_description.stencilLoadOp           = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
//		attachment_description.stencilStoreOp          = VK_ATTACHMENT_STORE_OP_DONT_CARE;
//		attachment_description.initialLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
//
//		switch (attachment.getType())
//		{
//			case Attachment::Type::Image:
//				attachment_description.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
//				attachment_description.format      = attachment.getFormat();
//				break;
//			case Attachment::Type::Depth:
//				attachment_description.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
//				attachment_description.format      = render_target.hasDepthAttachment() ? render_target.getDepthStencil()->getFormat() : VK_FORMAT_UNDEFINED;
//				break;
//			case Attachment::Type::Swapchain:
//				attachment_description.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
//				attachment_description.format      = GraphicsContext::instance()->getSurface().getFormat().format;
//				break;
//			default:
//				break;
//		}
//
//		attachment_descriptions.emplace_back(attachment_description);
//	}
//
//	// Create subpass and its dependencies
//	std::vector<VkSubpassDependency>                subpass_dependencies;
//	std::vector<VkSubpassDescription>               subpass_descriptions;
//	std::vector<std::vector<VkAttachmentReference>> output_attachments(render_target.getSubpasses().size());
//	std::vector<std::vector<VkAttachmentReference>> input_attachments(render_target.getSubpasses().size());
//	std::vector<VkAttachmentReference>              depth_attachments(render_target.getSubpasses().size());
//
//	uint32_t subpass_index = 0;
//
//	for (const auto &subpass : render_target.getSubpasses())
//	{
//		// Attachments
//		std::optional<uint32_t>            depth_attachment;
//		std::vector<VkAttachmentReference> subpass_input_attachments;
//		std::vector<VkAttachmentReference> subpass_output_attachments;
//
//		for (const auto &attachment_binding : subpass.getOutputAttachments())
//		{
//			auto attachment = render_target.getAttachment(attachment_binding);
//
//			if (!attachment)
//			{
//				VK_ERROR("Failed to find a renderpass attachment bound to: {}", attachment_binding);
//			}
//
//			if (attachment->getType() == Attachment::Type::Depth)
//			{
//				depth_attachment = attachment->getBinding();
//				continue;
//			}
//
//			VkAttachmentReference attachment_reference = {};
//			attachment_reference.attachment            = attachment->getBinding();
//			attachment_reference.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
//			subpass_output_attachments.push_back(attachment_reference);
//		}
//
//		for (const auto &attachment_binding : subpass.getInputAttachments())
//		{
//			auto attachment = render_target.getAttachment(attachment_binding);
//
//			if (!attachment)
//			{
//				VK_ERROR("Failed to find a renderpass attachment bound to: {}", attachment_binding);
//			}
//
//			VkAttachmentReference attachment_reference = {};
//			attachment_reference.attachment            = attachment->getBinding();
//			attachment_reference.layout                = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
//			subpass_input_attachments.push_back(attachment_reference);
//		}
//
//		output_attachments[subpass_index] = subpass_output_attachments;
//		input_attachments[subpass_index]  = subpass_input_attachments;
//
//		// Subpass Description
//		VkSubpassDescription subpass_description = {};
//		subpass_description.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
//		subpass_description.colorAttachmentCount = static_cast<uint32_t>(output_attachments[subpass_index].size());
//		subpass_description.pColorAttachments    = output_attachments[subpass_index].data();
//		subpass_description.inputAttachmentCount = static_cast<uint32_t>(input_attachments[subpass_index].size());
//		subpass_description.pInputAttachments    = input_attachments[subpass_index].data();
//
//		VkAttachmentReference depth_stencil_attachment = {};
//		if (depth_attachment)
//		{
//			depth_stencil_attachment.attachment         = depth_attachment.value();
//			depth_stencil_attachment.layout             = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
//			subpass_description.pDepthStencilAttachment = &depth_stencil_attachment;
//		}
//		subpass_descriptions.push_back(subpass_description);
//
//		// Subpass Dependencies
//		VkSubpassDependency subpass_dependency = {};
//		subpass_dependency.srcStageMask        = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
//		subpass_dependency.dstStageMask        = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
//		subpass_dependency.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
//		subpass_dependency.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
//		subpass_dependency.dependencyFlags     = VK_DEPENDENCY_BY_REGION_BIT;
//
//		if (subpass.getIndex() == render_target.getSubpasses().size())
//		{
//			subpass_dependency.dstSubpass    = VK_SUBPASS_EXTERNAL;
//			subpass_dependency.dstStageMask  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
//			subpass_dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
//			subpass_dependency.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
//		}
//		else
//		{
//			subpass_dependency.dstSubpass = subpass.getIndex();
//		}
//
//		if (subpass.getIndex() == 0)
//		{
//			subpass_dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
//			subpass_dependency.srcStageMask  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
//			subpass_dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
//			subpass_dependency.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
//			subpass_dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
//		}
//		else
//		{
//			subpass_dependency.srcSubpass = subpass.getIndex() - 1;
//		}
//
//		subpass_dependencies.emplace_back(subpass_dependency);
//		subpass_index++;
//	}
//
//	VkSubpassDependency subpass_dependency = {};
//	subpass_dependency.srcStageMask        = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
//	subpass_dependency.dstStageMask        = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
//	subpass_dependency.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
//	subpass_dependency.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
//	subpass_dependency.dependencyFlags     = VK_DEPENDENCY_BY_REGION_BIT;
//	subpass_dependency.dstSubpass          = VK_SUBPASS_EXTERNAL;
//	subpass_dependency.dstStageMask        = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
//	subpass_dependency.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
//	subpass_dependency.dstAccessMask       = VK_ACCESS_MEMORY_READ_BIT;
//	subpass_dependency.srcSubpass          = subpass_index - 1;
//	subpass_dependencies.emplace_back(subpass_dependency);
//
//	VkRenderPassCreateInfo render_pass_create_info = {};
//	render_pass_create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
//	render_pass_create_info.attachmentCount        = static_cast<uint32_t>(attachment_descriptions.size());
//	render_pass_create_info.pAttachments           = attachment_descriptions.data();
//	render_pass_create_info.subpassCount           = static_cast<uint32_t>(subpass_descriptions.size());
//	render_pass_create_info.pSubpasses             = subpass_descriptions.data();
//	render_pass_create_info.dependencyCount        = static_cast<uint32_t>(subpass_dependencies.size());
//	render_pass_create_info.pDependencies          = subpass_dependencies.data();
//
//	if (!VK_CHECK(vkCreateRenderPass(GraphicsContext::instance()->getLogicalDevice(), &render_pass_create_info, nullptr, &m_handle)))
//	{
//		VK_ERROR("Failed to create render pass!");
//	}
//}
//
//RenderPass::~RenderPass()
//{
//	vkDestroyRenderPass(GraphicsContext::instance()->getLogicalDevice(), m_handle, nullptr);
//}
//
//RenderPass::operator const VkRenderPass &() const
//{
//	return m_handle;
//}
//
//const VkRenderPass &RenderPass::getRenderPass() const
//{
//	return m_handle;
//}
//}        // namespace Ilum

#include "RenderPass.hpp"
#include "RenderGraph.hpp"

namespace Ilum
{
const Image &RenderPassState::getAttachment(const std::string &name)
{
	return graph.getAttachment(name);
}
}        // namespace Ilum