#include "PipelineAllocator.hpp"
#include "DescriptorAllocator.hpp"
#include "Device.hpp"

namespace Ilum
{
PipelineAllocator::PipelineAllocator(RHIDevice *device) :
    p_device(device)
{
}

PipelineAllocator::~PipelineAllocator()
{
	for (auto &[hash, pipeline] : m_pipelines)
	{
		vkDestroyPipeline(p_device->m_device, pipeline, nullptr);
	}
	for (auto &[hash, render_pass] : m_render_pass)
	{
		vkDestroyRenderPass(p_device->m_device, render_pass, nullptr);
	}
	for (auto &[hash, render_layout] : m_pipeline_layout)
	{
		vkDestroyPipelineLayout(p_device->m_device, render_layout, nullptr);
	}

	m_sbt.clear();
}

VkPipelineLayout PipelineAllocator::CreatePipelineLayout(PipelineState &pso)
{
	size_t hash = pso.Hash();
	if (m_pipeline_layout.find(hash) != m_pipeline_layout.end())
	{
		return m_pipeline_layout.at(hash);
	}

	m_pipeline_layout[hash] = VK_NULL_HANDLE;

	const auto &meta = pso.GetReflectionData();

	std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
	// Create descriptor layout & set
	for (auto &set : pso.GetReflectionData().sets)
	{
		descriptor_set_layouts.push_back(p_device->m_descriptor_allocator->GetDescriptorLayout(meta, set));
	}

	// Create pipeline layout
	std::vector<VkPushConstantRange> push_constants;
	for (auto &constant : meta.constants)
	{
		if (constant.type == ShaderReflectionData::Constant::Type::Push)
		{
			VkPushConstantRange push_constant_range = {};
			push_constant_range.stageFlags          = constant.stage;
			push_constant_range.size                = constant.size;
			push_constant_range.offset              = constant.offset;
			push_constants.push_back(push_constant_range);
		}
	}

	VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
	pipeline_layout_create_info.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeline_layout_create_info.pushConstantRangeCount     = static_cast<uint32_t>(push_constants.size());
	pipeline_layout_create_info.pPushConstantRanges        = push_constants.data();
	pipeline_layout_create_info.setLayoutCount             = static_cast<uint32_t>(descriptor_set_layouts.size());
	pipeline_layout_create_info.pSetLayouts                = descriptor_set_layouts.data();

	vkCreatePipelineLayout(p_device->m_device, &pipeline_layout_create_info, nullptr, &m_pipeline_layout[hash]);
	return m_pipeline_layout[hash];
}

VkRenderPass PipelineAllocator::CreateRenderPass(FrameBuffer &frame_buffer)
{
	size_t hash = frame_buffer.Hash();

	if (m_render_pass.find(hash) != m_render_pass.end())
	{
		return m_render_pass[hash];
	}

	m_render_pass[hash] = VK_NULL_HANDLE;

	VkSubpassDescription subpass_description    = {};
	subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass_description.colorAttachmentCount    = static_cast<uint32_t>(frame_buffer.m_attachment_references.size());
	subpass_description.pColorAttachments       = frame_buffer.m_attachment_references.data();
	subpass_description.pDepthStencilAttachment = frame_buffer.m_depth_stencil_attachment_reference.has_value() ? &frame_buffer.m_depth_stencil_attachment_reference.value() : nullptr;

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
	render_pass_create_info.attachmentCount        = static_cast<uint32_t>(frame_buffer.m_attachment_descriptions.size());
	render_pass_create_info.pAttachments           = frame_buffer.m_attachment_descriptions.data();
	render_pass_create_info.subpassCount           = 1;
	render_pass_create_info.pSubpasses             = &subpass_description;
	render_pass_create_info.dependencyCount        = static_cast<uint32_t>(subpass_dependencies.size());
	render_pass_create_info.pDependencies          = subpass_dependencies.data();

	vkCreateRenderPass(p_device->m_device, &render_pass_create_info, nullptr, &m_render_pass[hash]);

	return m_render_pass[hash];
}

VkFramebuffer PipelineAllocator::CreateFrameBuffer(FrameBuffer &frame_buffer)
{
	size_t hash = frame_buffer.Hash();

	if (m_frame_buffer.find(hash) != m_frame_buffer.end())
	{
		return m_frame_buffer[hash];
	}

	m_frame_buffer[hash] = VK_NULL_HANDLE;

	VkFramebufferCreateInfo frame_buffer_create_info = {};
	frame_buffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	frame_buffer_create_info.renderPass              = CreateRenderPass(frame_buffer);
	frame_buffer_create_info.attachmentCount         = static_cast<uint32_t>(frame_buffer.m_views.size());
	frame_buffer_create_info.pAttachments            = frame_buffer.m_views.data();
	frame_buffer_create_info.width                   = frame_buffer.m_width;
	frame_buffer_create_info.height                  = frame_buffer.m_height;
	frame_buffer_create_info.layers                  = frame_buffer.m_layer;

	vkCreateFramebuffer(p_device->m_device, &frame_buffer_create_info, nullptr, &m_frame_buffer[hash]);

	return m_frame_buffer[hash];
}

VkPipeline PipelineAllocator::CreatePipeline(PipelineState &pso)
{
	return VkPipeline();
}

}        // namespace Ilum