#include "RenderGraphBuilder.hpp"

#include "Graphics/Buffer/Buffer.h"
#include "Graphics/Descriptor/DescriptorBinding.hpp"
#include "Graphics/Descriptor/DescriptorCache.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Image/Image.hpp"
#include "Graphics/Pipeline/PipelineState.hpp"

#include "Device/LogicalDevice.hpp"
#include "Device/Swapchain.hpp"

#include "RenderGraph.hpp"
#include "RenderPass.hpp"

namespace Ilum
{
inline VkImageUsageFlagBits attachment_state_to_image_usage(AttachmentState state)
{
	switch (state)
	{
		case Ilum::AttachmentState::Discard_Color:
			return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		case Ilum::AttachmentState::Discard_Depth_Stencil:
			return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		case Ilum::AttachmentState::Load_Color:
			return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		case Ilum::AttachmentState::Load_Depth_Stencil:
			return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		case Ilum::AttachmentState::Clear_Color:
			return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		case Ilum::AttachmentState::Clear_Depth_Stencil:
			return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		default:
			return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	}
}

inline VkAttachmentLoadOp attachment_state_to_loadop(AttachmentState state)
{
	switch (state)
	{
		case Ilum::AttachmentState::Discard_Color:
			return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		case Ilum::AttachmentState::Discard_Depth_Stencil:
			return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		case Ilum::AttachmentState::Load_Color:
			return VK_ATTACHMENT_LOAD_OP_LOAD;
		case Ilum::AttachmentState::Load_Depth_Stencil:
			return VK_ATTACHMENT_LOAD_OP_LOAD;
		case Ilum::AttachmentState::Clear_Color:
			return VK_ATTACHMENT_LOAD_OP_CLEAR;
		case Ilum::AttachmentState::Clear_Depth_Stencil:
			return VK_ATTACHMENT_LOAD_OP_CLEAR;
		default:
			return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	}
}

inline VkAccessFlags buffer_usage_to_access(VkBufferUsageFlagBits usage)
{
	switch (usage)
	{
		case VK_BUFFER_USAGE_TRANSFER_SRC_BIT:
			return VK_ACCESS_TRANSFER_READ_BIT;
		case VK_BUFFER_USAGE_TRANSFER_DST_BIT:
			return VK_ACCESS_TRANSFER_WRITE_BIT;
		case VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT:
			return VK_ACCESS_SHADER_READ_BIT;
		case VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT:
			return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		case VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT:
			return VK_ACCESS_SHADER_READ_BIT;
		case VK_BUFFER_USAGE_STORAGE_BUFFER_BIT:
			return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		case VK_BUFFER_USAGE_INDEX_BUFFER_BIT:
			return VK_ACCESS_INDEX_READ_BIT;
		case VK_BUFFER_USAGE_VERTEX_BUFFER_BIT:
			return VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
		case VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT:
			return VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
		case VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT:
			return VK_ACCESS_SHADER_READ_BIT;
		case VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT:
			return VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT;
		case VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT:
			return VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT;
		case VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT:
			return VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT;
		case VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR:
			return VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
		case VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR:
			return VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
		case VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR:
			return VK_ACCESS_SHADER_READ_BIT;
		default:
			return VK_ACCESS_FLAG_BITS_MAX_ENUM;
	}
}

inline bool hasImageWriteDependency(VkImageUsageFlagBits usage)
{
	switch (usage)
	{
		case VK_IMAGE_USAGE_TRANSFER_DST_BIT:
		case VK_IMAGE_USAGE_STORAGE_BIT:
		case VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT:
		case VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT:
		case VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR:
			return true;
		default:
			return false;
	}
}

inline bool hasBufferWriteDependency(VkBufferUsageFlagBits usage)
{
	switch (usage)
	{
		case VK_BUFFER_USAGE_TRANSFER_DST_BIT:
		case VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT:
		case VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT:
		case VK_BUFFER_USAGE_STORAGE_BUFFER_BIT:
		case VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT:
		case VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT:
		case VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR:
			return true;
		default:
			return false;
	}
}

inline VkPipelineStageFlags buffer_usage_to_stage(VkBufferUsageFlagBits usage)
{
	switch (usage)
	{
		case VK_BUFFER_USAGE_TRANSFER_SRC_BIT:
			return VK_PIPELINE_STAGE_TRANSFER_BIT;
		case VK_BUFFER_USAGE_TRANSFER_DST_BIT:
			return VK_PIPELINE_STAGE_TRANSFER_BIT;
		case VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT:
			return VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
		case VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT:
			return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		case VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT:
			return VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
		case VK_BUFFER_USAGE_STORAGE_BUFFER_BIT:
			return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		case VK_BUFFER_USAGE_INDEX_BUFFER_BIT:
			return VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
		case VK_BUFFER_USAGE_VERTEX_BUFFER_BIT:
			return VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
		case VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT:
			return VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
		case VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT:
			return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		case VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT:
			return VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
		case VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT:
			return VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
		case VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT:
			return VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT;
		case VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR:
			return VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
		case VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR:
			return VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
		case VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR:
			return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		case VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM:
			return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		default:
			return VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
	}
}

inline void createGraphicsPipeline(const PipelineState &pipeline_state, PassNative &pass_native)
{
	// Input Assembly State
	VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info = {};
	input_assembly_state_create_info.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	input_assembly_state_create_info.topology                               = pipeline_state.input_assembly_state.topology;
	input_assembly_state_create_info.flags                                  = 0;
	input_assembly_state_create_info.primitiveRestartEnable                 = pipeline_state.input_assembly_state.primitive_restart_enable;

	// Rasterization State
	VkPipelineRasterizationStateCreateInfo rasterization_state_create_info = {};
	rasterization_state_create_info.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterization_state_create_info.polygonMode                            = pipeline_state.rasterization_state.polygon_mode;
	rasterization_state_create_info.cullMode                               = pipeline_state.rasterization_state.cull_mode;
	rasterization_state_create_info.frontFace                              = pipeline_state.rasterization_state.front_face;
	rasterization_state_create_info.flags                                  = 0;
	rasterization_state_create_info.depthClampEnable                       = VK_FALSE;
	rasterization_state_create_info.lineWidth                              = 1.0f;

	// Color Blend Attachment State
	std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachment_states(pipeline_state.color_blend_attachment_states.size());

	for (uint32_t i = 0; i < color_blend_attachment_states.size(); i++)
	{
		color_blend_attachment_states[i].blendEnable         = pipeline_state.color_blend_attachment_states[i].blend_enable;
		color_blend_attachment_states[i].srcColorBlendFactor = pipeline_state.color_blend_attachment_states[i].src_color_blend_factor;
		color_blend_attachment_states[i].dstColorBlendFactor = pipeline_state.color_blend_attachment_states[i].dst_color_blend_factor;
		color_blend_attachment_states[i].colorBlendOp        = pipeline_state.color_blend_attachment_states[i].color_blend_op;
		color_blend_attachment_states[i].srcAlphaBlendFactor = pipeline_state.color_blend_attachment_states[i].src_alpha_blend_factor;
		color_blend_attachment_states[i].dstAlphaBlendFactor = pipeline_state.color_blend_attachment_states[i].dst_alpha_blend_factor;
		color_blend_attachment_states[i].alphaBlendOp        = pipeline_state.color_blend_attachment_states[i].alpha_blend_op;
		color_blend_attachment_states[i].colorWriteMask      = pipeline_state.color_blend_attachment_states[i].color_write_mask;
	}

	VkPipelineColorBlendStateCreateInfo color_blend_state_create_info = {};
	color_blend_state_create_info.sType                               = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blend_state_create_info.logicOpEnable                       = VK_FALSE;
	color_blend_state_create_info.logicOp                             = VK_LOGIC_OP_COPY;
	color_blend_state_create_info.attachmentCount                     = static_cast<uint32_t>(color_blend_attachment_states.size());
	color_blend_state_create_info.pAttachments                        = color_blend_attachment_states.data();
	color_blend_state_create_info.blendConstants[0]                   = 0.0f;
	color_blend_state_create_info.blendConstants[1]                   = 0.0f;
	color_blend_state_create_info.blendConstants[2]                   = 0.0f;
	color_blend_state_create_info.blendConstants[3]                   = 0.0f;

	// Depth Stencil State
	VkPipelineDepthStencilStateCreateInfo depth_stencil_state_create_info = {};
	depth_stencil_state_create_info.sType                                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth_stencil_state_create_info.depthTestEnable                       = pipeline_state.depth_stencil_state.depth_test_enable;
	depth_stencil_state_create_info.depthWriteEnable                      = pipeline_state.depth_stencil_state.depth_write_enable;
	depth_stencil_state_create_info.depthCompareOp                        = pipeline_state.depth_stencil_state.depth_compare_op;
	depth_stencil_state_create_info.back.compareOp                        = VK_COMPARE_OP_ALWAYS;

	// Viewport State
	VkPipelineViewportStateCreateInfo viewport_state_create_info = {};
	viewport_state_create_info.sType                             = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport_state_create_info.viewportCount                     = pipeline_state.viewport_state.viewport_count;
	viewport_state_create_info.scissorCount                      = pipeline_state.viewport_state.scissor_count;
	viewport_state_create_info.flags                             = 0;

	// Multisample State
	VkPipelineMultisampleStateCreateInfo multisample_state_create_info = {};
	multisample_state_create_info.sType                                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample_state_create_info.rasterizationSamples                 = pipeline_state.multisample_state.sample_count;
	multisample_state_create_info.flags                                = 0;

	// Dynamic State
	VkPipelineDynamicStateCreateInfo dynamic_state_create_info = {};
	dynamic_state_create_info.sType                            = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic_state_create_info.pDynamicStates                   = pipeline_state.dynamic_state.dynamic_states.data();
	dynamic_state_create_info.dynamicStateCount                = static_cast<uint32_t>(pipeline_state.dynamic_state.dynamic_states.size());
	dynamic_state_create_info.flags                            = 0;

	std::vector<VkPipelineShaderStageCreateInfo> shader_module_create_infos;
	for (auto &[stage, shader] : pipeline_state.shader.getShaders())
	{
		VkPipelineShaderStageCreateInfo shader_stage_create_info = {};

		shader_stage_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shader_stage_create_info.stage  = stage;
		shader_stage_create_info.pName  = "main";
		shader_stage_create_info.module = shader;
		shader_module_create_infos.push_back(shader_stage_create_info);
	}

	// Vertex Input State
	VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info = {};
	vertex_input_state_create_info.sType                                = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertex_input_state_create_info.vertexAttributeDescriptionCount      = static_cast<uint32_t>(pipeline_state.vertex_input_state.attribute_descriptions.size());
	vertex_input_state_create_info.pVertexAttributeDescriptions         = pipeline_state.vertex_input_state.attribute_descriptions.data();
	vertex_input_state_create_info.vertexBindingDescriptionCount        = static_cast<uint32_t>(pipeline_state.vertex_input_state.binding_descriptions.size());
	vertex_input_state_create_info.pVertexBindingDescriptions           = pipeline_state.vertex_input_state.binding_descriptions.data();

	VkGraphicsPipelineCreateInfo graphics_pipeline_create_info = {};
	graphics_pipeline_create_info.sType                        = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	graphics_pipeline_create_info.stageCount                   = static_cast<uint32_t>(shader_module_create_infos.size());
	graphics_pipeline_create_info.pStages                      = shader_module_create_infos.data();

	graphics_pipeline_create_info.pInputAssemblyState = &input_assembly_state_create_info;
	graphics_pipeline_create_info.pRasterizationState = &rasterization_state_create_info;
	graphics_pipeline_create_info.pColorBlendState    = &color_blend_state_create_info;
	graphics_pipeline_create_info.pDepthStencilState  = &depth_stencil_state_create_info;
	graphics_pipeline_create_info.pViewportState      = &viewport_state_create_info;
	graphics_pipeline_create_info.pMultisampleState   = &multisample_state_create_info;
	graphics_pipeline_create_info.pDynamicState       = &dynamic_state_create_info;
	graphics_pipeline_create_info.pVertexInputState   = &vertex_input_state_create_info;

	graphics_pipeline_create_info.layout             = pass_native.pipeline_layout;
	graphics_pipeline_create_info.renderPass         = pass_native.render_pass;
	graphics_pipeline_create_info.subpass            = 0;
	graphics_pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
	graphics_pipeline_create_info.basePipelineIndex  = -1;

	vkCreateGraphicsPipelines(GraphicsContext::instance()->getLogicalDevice(), GraphicsContext::instance()->getPipelineCache(), 1, &graphics_pipeline_create_info, nullptr, &pass_native.pipeline);
}

inline void createComputePipeline(const PipelineState &pipeline_state, PassNative &pass_native)
{
	VkPipelineShaderStageCreateInfo shader_stage_create_info = {};
	shader_stage_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shader_stage_create_info.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	shader_stage_create_info.pName  = "main";
	shader_stage_create_info.module = pipeline_state.shader.getShaders().at(VK_SHADER_STAGE_COMPUTE_BIT);

	VkComputePipelineCreateInfo compute_pipeline_create_info = {};
	compute_pipeline_create_info.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	compute_pipeline_create_info.stage                       = shader_stage_create_info;
	compute_pipeline_create_info.layout                      = pass_native.pipeline_layout;
	compute_pipeline_create_info.basePipelineIndex           = 0;
	compute_pipeline_create_info.basePipelineHandle          = VK_NULL_HANDLE;

	vkCreateComputePipelines(GraphicsContext::instance()->getLogicalDevice(), GraphicsContext::instance()->getPipelineCache(), 1, &compute_pipeline_create_info, nullptr, &pass_native.pipeline);
}

inline VkImageMemoryBarrier createImageMemoryBarrier(VkImage image, VkImageUsageFlagBits old_usage, VkImageUsageFlagBits new_usage, VkFormat format, uint32_t mip_level_count, uint32_t layer_count)
{
	VkImageSubresourceRange subresource_range = {};
	subresource_range.aspectMask              = Image::format_to_aspect(format);
	subresource_range.baseArrayLayer          = 0;
	subresource_range.baseMipLevel            = 0;
	subresource_range.layerCount              = layer_count;
	subresource_range.levelCount              = mip_level_count;

	VkImageMemoryBarrier barrier = {};
	barrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.image                = image;
	barrier.oldLayout            = Image::usage_to_layout(old_usage);
	barrier.newLayout            = Image::usage_to_layout(new_usage);
	barrier.srcAccessMask        = Image::usage_to_access(old_usage);
	barrier.dstAccessMask        = Image::usage_to_access(new_usage);
	barrier.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
	barrier.subresourceRange     = subresource_range;

	return barrier;
}

inline VkBufferMemoryBarrier createBufferMemoryBarrier(VkBuffer buffer, VkBufferUsageFlagBits old_usage, VkBufferUsageFlagBits new_usage)
{
	VkBufferMemoryBarrier barrier = {};
	barrier.sType                 = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	barrier.buffer                = buffer;
	barrier.srcAccessMask         = buffer_usage_to_access(old_usage);
	barrier.dstAccessMask         = buffer_usage_to_access(new_usage);
	barrier.srcQueueFamilyIndex   = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex   = VK_QUEUE_FAMILY_IGNORED;
	barrier.size                  = VK_WHOLE_SIZE;
	barrier.offset                = 0;

	return barrier;
}

inline void insertPipelineBarrier(const CommandBuffer &command_buffer, const ResolveInfo &resolve_info, const std::unordered_map<std::string, BufferTransition> &buffer_transitions, const std::unordered_map<std::string, ImageTransition> &image_transitions)
{
	VkPipelineStageFlags src_pipeline_flags = {};
	VkPipelineStageFlags dst_pipeline_flags = {};

	std::vector<VkBufferMemoryBarrier> buffer_barriers;
	std::vector<VkImageMemoryBarrier>  image_barriers;

	// Insert buffer barrier
	for (const auto &[buffer_name, buffer_transition] : buffer_transitions)
	{
		if (!hasBufferWriteDependency(buffer_transition.initial_usage))
		{
			continue;
		}

		src_pipeline_flags |= buffer_usage_to_stage(buffer_transition.initial_usage);
		dst_pipeline_flags |= buffer_usage_to_stage(buffer_transition.final_usage);

		auto buffers = resolve_info.getBuffers().at(buffer_name);
		for (const auto &buffer : buffers)
		{
			buffer_barriers.push_back(createBufferMemoryBarrier(buffer.get().getBuffer(), buffer_transition.initial_usage, buffer_transition.final_usage));
		}
	}

	// Insert image barrier
	for (const auto &[image_name, image_transition] : image_transitions)
	{
		if (image_transition.initial_usage == image_transition.final_usage && !hasImageWriteDependency(image_transition.initial_usage))
		{
			continue;
		}

		src_pipeline_flags |= Image::usage_to_stage(image_transition.initial_usage);
		dst_pipeline_flags |= Image::usage_to_stage(image_transition.final_usage);

		const auto &images = resolve_info.getImages().at(image_name);
		for (const auto &image : images)
		{
			image_barriers.push_back(createImageMemoryBarrier(image.get().getImage(), image_transition.initial_usage, image_transition.final_usage, image.get().getFormat(), image.get().getMipLevelCount(), image.get().getLayerCount()));
		}
	}

	if (buffer_barriers.empty() && image_barriers.empty())
	{
		return;
	}

	vkCmdPipelineBarrier(command_buffer, src_pipeline_flags, dst_pipeline_flags, 0, 0, nullptr, static_cast<uint32_t>(buffer_barriers.size()), buffer_barriers.data(), static_cast<uint32_t>(image_barriers.size()), image_barriers.data());
}

inline VkSemaphore createSemaphore()
{
	VkSemaphore           semaphore = VK_NULL_HANDLE;
	VkSemaphoreCreateInfo create_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
	create_info.pNext = nullptr;
	create_info.flags = 0;

	vkCreateSemaphore(GraphicsContext::instance()->getLogicalDevice(), &create_info, nullptr, &semaphore);

	return semaphore;
}

RenderGraphBuilder &RenderGraphBuilder::addRenderPass(const std::string &name, std::unique_ptr<RenderPass> render_pass)
{
	m_render_pass_references.push_back({name, std::move(render_pass)});
	return *this;
}

RenderGraphBuilder &RenderGraphBuilder::setOutput(const std::string &name)
{
	m_output = name;
	return *this;
}

RenderGraphBuilder &RenderGraphBuilder::setView(const std::string &name)
{
	m_view = name;
	return *this;
}

scope<RenderGraph> RenderGraphBuilder::build()
{
	if (m_render_pass_references.empty())
	{
		return createScope<RenderGraph>();
	}

	// Prepare:
	// - Create pipeline states
	auto pipeline_states = createPipelineStates();

	// - Create synchronize dependency
	// TODO: Fixing multi-threading rendering
	//auto synchronize_dependency = createSynchronizeDependency(pipeline_states);

	// - Resolve resource transitions
	auto resource_transitions = resolveResourceTransitions(pipeline_states);

	// - Setup output image
	if (!m_output.empty())
	{
		setOutputImage(resource_transitions, m_output);
	}
	// - Allocate attachments
	auto attachments = allocateAttachments(pipeline_states, resource_transitions);

	// Build render pass
	std::vector<RenderGraphNode> nodes;

	for (auto &render_pass_reference : m_render_pass_references)
	{
		auto render_pass = buildRenderPass(render_pass_reference, pipeline_states, attachments, resource_transitions);

		nodes.push_back(RenderGraphNode{
		    render_pass_reference.name,
		    render_pass,
		    std::move(render_pass_reference.pass),
		    getRenderPassAttachmentNames(render_pass_reference.name, pipeline_states),
		    createPipelineBarrierCallback(render_pass_reference.name, pipeline_states.at(render_pass_reference.name), resource_transitions),
		    pipeline_states.at(render_pass_reference.name).descriptor_bindings/*,
		    synchronize_dependency.at(render_pass_reference.name)*/});
	}

	return createScope<RenderGraph>(
	    std::move(nodes),
	    std::move(attachments),
	    m_output,
	    m_view,
	    m_output.empty() ? [](const CommandBuffer &, const Image &, const Image &) {} : createOnPresentCallback(m_output, resource_transitions),
	    createOnCreateCallback(pipeline_states, resource_transitions, attachments));
}

const std::string &RenderGraphBuilder::output() const
{
	return m_output;
}

const std::string &RenderGraphBuilder::view() const
{
	return m_view;
}

void RenderGraphBuilder::reset()
{
	m_output = "output";
	m_view   = "view";
	m_render_pass_references.clear();
}

bool RenderGraphBuilder::empty() const
{
	return m_render_pass_references.empty();
}

RenderGraphBuilder::PipelineMap RenderGraphBuilder::createPipelineStates()
{
	PipelineMap pipeline_states;
	for (const auto &render_pass_reference : m_render_pass_references)
	{
		auto &pipeline = pipeline_states[render_pass_reference.name];
		render_pass_reference.pass->setupPipeline(pipeline);

		for (const auto &[set, buffers] : pipeline.descriptor_bindings.getBoundBuffers())
		{
			for (const auto &buffer : buffers)
			{
				pipeline.addDependency(buffer.name, buffer.usage);
			}
		}

		for (const auto &[set, images] : pipeline.descriptor_bindings.getBoundImages())
		{
			for (const auto &image : images)
			{
				pipeline.addDependency(image.name, image.usage);
			}
		}
	}

	return pipeline_states;
}

RenderGraphBuilder::ResourceTransitions RenderGraphBuilder::resolveResourceTransitions(const PipelineMap &pipeline_states)
{
	ResourceTransitions                                    resource_transitions;
	std::unordered_map<std::string, VkBufferUsageFlagBits> last_buffer_usages;
	std::unordered_map<std::string, VkImageUsageFlagBits>  last_image_usages;

	for (const auto &render_pass_reference : m_render_pass_references)
	{
		auto &pipeline_state     = pipeline_states.at(render_pass_reference.name);
		auto &buffer_transitions = resource_transitions.buffers.transitions[render_pass_reference.name];
		auto &image_transitions  = resource_transitions.images.transitions[render_pass_reference.name];

		// Handle buffer dependency
		for (const auto &buffer_dependency : pipeline_state.getBufferDependencies())
		{
			if (last_buffer_usages.find(buffer_dependency.name) == last_buffer_usages.end())
			{
				resource_transitions.buffers.first_usages[buffer_dependency.name] = render_pass_reference.name;
				last_buffer_usages[buffer_dependency.name]                        = VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM;
			}
			buffer_transitions[buffer_dependency.name].initial_usage = last_buffer_usages.at(buffer_dependency.name);
			buffer_transitions[buffer_dependency.name].final_usage   = buffer_dependency.usage;
			resource_transitions.buffers.total_usages[buffer_dependency.name] |= buffer_dependency.usage;
			last_buffer_usages[buffer_dependency.name]                       = buffer_dependency.usage;
			resource_transitions.buffers.last_usages[buffer_dependency.name] = render_pass_reference.name;
		}

		// Handle image dependency
		for (const auto &image_dependency : pipeline_state.getImageDependencies())
		{
			if (last_image_usages.find(image_dependency.name) == last_image_usages.end())
			{
				resource_transitions.images.first_usages[image_dependency.name] = render_pass_reference.name;
				last_image_usages[image_dependency.name]                        = VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM;
			}

			image_transitions[image_dependency.name].initial_usage = last_image_usages.at(image_dependency.name);
			image_transitions[image_dependency.name].final_usage   = image_dependency.usage;
			resource_transitions.images.total_usages[image_dependency.name] |= image_dependency.usage;
			last_image_usages[image_dependency.name]                       = image_dependency.usage;
			resource_transitions.images.last_usages[image_dependency.name] = render_pass_reference.name;
		}

		// Handle attachments
		for (const auto &attachment_dependency : pipeline_state.getOutputAttachments())
		{
			auto attachment_dependency_usage = attachment_state_to_image_usage(attachment_dependency.state);

			if (last_image_usages.find(attachment_dependency.name) == last_image_usages.end())
			{
				resource_transitions.images.first_usages[attachment_dependency.name] = render_pass_reference.name;
				last_image_usages[attachment_dependency.name]                        = VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM;
			}

			image_transitions[attachment_dependency.name].initial_usage = last_image_usages.at(attachment_dependency.name);
			image_transitions[attachment_dependency.name].final_usage   = attachment_dependency_usage;
			resource_transitions.images.total_usages[attachment_dependency.name] |= attachment_dependency_usage;
			last_image_usages[attachment_dependency.name]                       = attachment_dependency_usage;
			resource_transitions.images.last_usages[attachment_dependency.name] = render_pass_reference.name;
		}
	}

	// Setting first usage using last second usage
	for (const auto &[buffer_name, pass_name] : resource_transitions.buffers.first_usages)
	{
		resource_transitions.buffers.transitions[pass_name][buffer_name].initial_usage = last_buffer_usages[buffer_name];
	}
	for (const auto &[image_name, pass_name] : resource_transitions.images.first_usages)
	{
		resource_transitions.images.transitions[pass_name][image_name].initial_usage = last_image_usages[image_name];
	}

	return resource_transitions;
}

RenderGraphBuilder::SynchronizeMap RenderGraphBuilder::createSynchronizeDependency(const PipelineMap &pipeline_states)
{
	SynchronizeMap                            synchronize_map;
	std::unordered_map<std::string, uint32_t> pass_order;
	for (uint32_t i = 0; i < m_render_pass_references.size(); i++)
	{
		pass_order[m_render_pass_references[i].name] = i;
	}

	std::unordered_map<std::string, std::vector<std::string>> read_image_map;
	std::unordered_map<std::string, std::vector<std::string>> read_buffer_map;
	std::unordered_map<std::string, std::vector<std::string>> write_image_map;
	std::unordered_map<std::string, std::vector<std::string>> write_buffer_map;

	for (auto &[pass_name, pipeline_state] : pipeline_states)
	{
		// Handle buffer
		for (auto &buffer_dependency : pipeline_state.getBufferDependencies())
		{
			if (hasBufferWriteDependency(buffer_dependency.usage))
			{
				write_buffer_map[buffer_dependency.name].push_back(pass_name);
			}
			else
			{
				read_buffer_map[buffer_dependency.name].push_back(pass_name);
			}
		}

		// Handle image
		for (auto &image_dependency : pipeline_state.getImageDependencies())
		{
			if (hasImageWriteDependency(image_dependency.usage))
			{
				write_image_map[image_dependency.name].push_back(pass_name);
			}
			else
			{
				read_image_map[image_dependency.name].push_back(pass_name);
			}
		}

		// Handle attachment
		for (auto &attachment_dependency : pipeline_state.getOutputAttachments())
		{
			write_image_map[attachment_dependency.name].push_back(pass_name);
		}



		VkFence           fence             = VK_NULL_HANDLE;
		VkFenceCreateInfo fence_create_info = {};
		fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_create_info.flags             = 0;

		vkCreateFence(GraphicsContext::instance()->getLogicalDevice(), &fence_create_info, nullptr, &fence);

		SubmitInfo submit_info;
		submit_info.fence            = fence;

		synchronize_map.insert({pass_name, submit_info});
	}

	for (auto &[pass_name, pipeline_state] : pipeline_states)
	{
		// Handle buffer
		for (auto &buffer_dependency : pipeline_state.getBufferDependencies())
		{
			if (hasBufferWriteDependency(buffer_dependency.usage))
			{
				if (read_buffer_map.find(buffer_dependency.name) == read_buffer_map.end())
				{
					continue;
				}

				for (auto &pass : read_buffer_map[buffer_dependency.name])
				{
					if (pass_order.at(pass) < pass_order.at(pass_name))
					{
						bool found = false;
						for (auto &semaphore : synchronize_map[pass_name].wait_semaphores)
						{
							if (semaphore == synchronize_map[pass].signal_semaphore)
							{
								found = true;
							}
						}

						if (!found)
						{
							if (!synchronize_map[pass].signal_semaphore)
							{
								synchronize_map[pass].signal_semaphore = createSemaphore();
							}

							synchronize_map[pass_name].wait_semaphores.push_back(synchronize_map[pass].signal_semaphore);
							synchronize_map[pass_name].wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
						}
					}
				}
			}
			else
			{
				if (write_buffer_map.find(buffer_dependency.name) == write_buffer_map.end())
				{
					continue;
				}

				for (auto &pass : write_buffer_map[buffer_dependency.name])
				{
					if (pass_order.at(pass) < pass_order.at(pass_name))
					{
						bool found = false;
						for (auto &semaphore : synchronize_map[pass_name].wait_semaphores)
						{
							if (semaphore == synchronize_map[pass].signal_semaphore)
							{
								found = true;
							}
						}

						if (!found)
						{
							if (!synchronize_map[pass].signal_semaphore)
							{
								synchronize_map[pass].signal_semaphore = createSemaphore();
							}
							synchronize_map[pass_name].wait_semaphores.push_back(synchronize_map[pass].signal_semaphore);
							synchronize_map[pass_name].wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
						}
					}
				}
			}
		}

		// Handle image
		for (auto &image_dependency : pipeline_state.getImageDependencies())
		{
			if (hasImageWriteDependency(image_dependency.usage))
			{
				if (read_image_map.find(image_dependency.name) == read_image_map.end())
				{
					continue;
				}

				// Only one pass can write into it at the same time
				for (auto &pass : read_image_map[image_dependency.name])
				{
					if (pass_order.at(pass) < pass_order.at(pass_name))
					{
						bool found = false;
						for (auto &semaphore : synchronize_map[pass_name].wait_semaphores)
						{
							if (semaphore == synchronize_map[pass].signal_semaphore)
							{
								found = true;
							}
						}

						if (!found)
						{
							if (!synchronize_map[pass].signal_semaphore)
							{
								synchronize_map[pass].signal_semaphore = createSemaphore();
							}
							synchronize_map[pass_name].wait_semaphores.push_back(synchronize_map[pass].signal_semaphore);
							synchronize_map[pass_name].wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
						}
					}
				}
				for (auto &pass : write_image_map[image_dependency.name])
				{
					if (pass_order.at(pass) < pass_order.at(pass_name))
					{
						bool found = false;
						for (auto &semaphore : synchronize_map[pass_name].wait_semaphores)
						{
							if (semaphore == synchronize_map[pass].signal_semaphore)
							{
								found = true;
							}
						}

						if (!found)
						{
							if (!synchronize_map[pass].signal_semaphore)
							{
								synchronize_map[pass].signal_semaphore = createSemaphore();
							}
							synchronize_map[pass_name].wait_semaphores.push_back(synchronize_map[pass].signal_semaphore);
							synchronize_map[pass_name].wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
						}
					}
				}
			}
			else
			{
				if (write_image_map.find(image_dependency.name) == write_image_map.end())
				{
					continue;
				}

				for (auto &pass : write_image_map[image_dependency.name])
				{
					if (pass_order.at(pass) < pass_order.at(pass_name))
					{
						bool found = false;
						for (auto &semaphore : synchronize_map[pass_name].wait_semaphores)
						{
							if (semaphore == synchronize_map[pass].signal_semaphore)
							{
								found = true;
							}
						}

						if (!found)
						{
							if (!synchronize_map[pass].signal_semaphore)
							{
								synchronize_map[pass].signal_semaphore = createSemaphore();
							}
							synchronize_map[pass_name].wait_semaphores.push_back(synchronize_map[pass].signal_semaphore);
							synchronize_map[pass_name].wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
						}
					}
				}
			}
		}

		// Handle attachment
		for (auto &attachment_dependency : pipeline_state.getOutputAttachments())
		{
			if (read_image_map.find(attachment_dependency.name) == read_image_map.end())
			{
				continue;
			}

			for (auto &pass : read_image_map[attachment_dependency.name])
			{
				if (pass_order.at(pass) < pass_order.at(pass_name))
				{
					bool found = false;
					for (auto &semaphore : synchronize_map[pass_name].wait_semaphores)
					{
						if (semaphore == synchronize_map[pass].signal_semaphore)
						{
							found = true;
						}
					}

					if (!found)
					{
						if (!synchronize_map[pass].signal_semaphore)
						{
							synchronize_map[pass].signal_semaphore = createSemaphore();
						}
						synchronize_map[pass_name].wait_semaphores.push_back(synchronize_map[pass].signal_semaphore);
						synchronize_map[pass_name].wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
					}
				}
			}
		}
	}

	return synchronize_map;
}

void RenderGraphBuilder::setOutputImage(ResourceTransitions &resource_transitions, const std::string &name)
{
	resource_transitions.images.total_usages.at(name) |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
}

RenderGraphBuilder::AttachmentMap RenderGraphBuilder::allocateAttachments(const PipelineMap &pipeline_states, const ResourceTransitions &resource_transitions)
{
	AttachmentMap attachments;

	auto [surface_width, surface_height] = GraphicsContext::instance()->getSwapchain().getExtent();

	for (const auto &[pass_name, pipeline_state] : pipeline_states)
	{
		for (const auto &attachment : pipeline_state.getAttachmentDeclarations())
		{
			auto attachment_usage = resource_transitions.images.total_usages.at(attachment.name);
			attachments.emplace(attachment.name, std::move(Image(
			                                         attachment.width == 0 ? surface_width : attachment.width,
			                                         attachment.height == 0 ? surface_height : attachment.height,
			                                         attachment.format,
			                                         attachment_usage | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
			                                         VMA_MEMORY_USAGE_GPU_ONLY,
			                                         attachment.mipmaps,
			                                         attachment.layers)));
		}
	}
	return attachments;
}

PassNative RenderGraphBuilder::buildRenderPass(const RenderPassReference &render_pass_reference, const PipelineMap &pipeline_states, const AttachmentMap &attachments, const ResourceTransitions &resource_transitions)
{
	PassNative pass_native;

	std::vector<VkAttachmentDescription> attachment_descriptions;
	std::vector<VkAttachmentReference>   attachment_references;
	std::vector<VkImageView>             attachment_views;

	VkExtent2D render_area = {0, 0};

	auto &pass                    = pipeline_states.at(render_pass_reference.name);
	auto &render_pass_attachments = pass.getOutputAttachments();
	auto &image_transitions       = resource_transitions.images.transitions.at(render_pass_reference.name);

	// Only for graphics pipeline
	if (!render_pass_attachments.empty())
	{
		std::optional<VkAttachmentReference> depth_stencil_attachment_reference;
		for (uint32_t attachment_index = 0; attachment_index < render_pass_attachments.size(); attachment_index++)
		{
			const auto &attachment            = render_pass_attachments[attachment_index];
			const auto &image_reference       = attachments.at(attachment.name);
			const auto &attachment_transition = image_transitions.at(attachment.name);

			if (render_area.width == 0 && render_area.height == 0)
			{
				render_area.width  = std::max(render_area.width, image_reference.getWidth());
				render_area.height = std::max(render_area.height, image_reference.getHeight());
			}

			VkAttachmentDescription attachment_description = {};
			attachment_description.format                  = image_reference.getFormat();
			attachment_description.samples                 = VK_SAMPLE_COUNT_1_BIT;
			attachment_description.loadOp                  = attachment_state_to_loadop(attachment.state);
			attachment_description.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
			attachment_description.stencilLoadOp           = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			attachment_description.stencilStoreOp          = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			attachment_description.initialLayout           = Image::usage_to_layout(attachment_transition.final_usage);
			attachment_description.finalLayout             = Image::usage_to_layout(attachment_transition.final_usage);

			attachment_descriptions.push_back(std::move(attachment_description));
			if (attachment.layer == PipelineState::OutputAttachment::ALL_LAYERS)
			{
				attachment_views.push_back(image_reference.getView(ImageViewType::Native));
			}
			else
			{
				attachment_views.push_back(image_reference.getView(attachment.layer, ImageViewType::Native));
			}

			VkAttachmentReference attachment_reference = {};
			attachment_reference.attachment            = attachment_index;
			attachment_reference.layout                = Image::usage_to_layout(attachment_transition.final_usage);

			if (attachment_transition.final_usage == VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
			{
				depth_stencil_attachment_reference = std::move(attachment_reference);
				VkClearValue clear                 = {};
				clear.depthStencil                 = attachment.depth_stencil_clear;
				pass_native.clear_values.push_back(clear);
			}
			else
			{
				attachment_references.push_back(std::move(attachment_reference));
				VkClearValue clear = {};
				clear.color        = attachment.color_clear;
				pass_native.clear_values.push_back(clear);
			}
		}

		// TODO: Subpasses?
		VkSubpassDescription subpass_description    = {};
		subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass_description.colorAttachmentCount    = static_cast<uint32_t>(attachment_references.size());
		subpass_description.pColorAttachments       = attachment_references.data();
		subpass_description.pDepthStencilAttachment = depth_stencil_attachment_reference.has_value() ? &depth_stencil_attachment_reference.value() : nullptr;

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
		render_pass_create_info.attachmentCount        = static_cast<uint32_t>(attachment_descriptions.size());
		render_pass_create_info.pAttachments           = attachment_descriptions.data();
		render_pass_create_info.subpassCount           = 1;
		render_pass_create_info.pSubpasses             = &subpass_description;
		render_pass_create_info.dependencyCount        = static_cast<uint32_t>(subpass_dependencies.size());
		render_pass_create_info.pDependencies          = subpass_dependencies.data();

		vkCreateRenderPass(GraphicsContext::instance()->getLogicalDevice(), &render_pass_create_info, nullptr, &pass_native.render_pass);

		// Create framebuffer
		VkFramebufferCreateInfo frame_buffer_create_info = {};
		frame_buffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frame_buffer_create_info.renderPass              = pass_native.render_pass;
		frame_buffer_create_info.attachmentCount         = static_cast<uint32_t>(attachment_views.size());
		frame_buffer_create_info.pAttachments            = attachment_views.data();
		frame_buffer_create_info.width                   = render_area.width;
		frame_buffer_create_info.height                  = render_area.height;
		frame_buffer_create_info.layers                  = 1;

		vkCreateFramebuffer(GraphicsContext::instance()->getLogicalDevice(), &frame_buffer_create_info, nullptr, &pass_native.frame_buffer);

		pass_native.render_area.extent = render_area;
		pass_native.render_area.offset = {0, 0};
	}

	pass_native.bind_point = pass.shader.getBindPoint();

	std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
	// Create descriptor layout & set
	for (auto &set : pass.shader.getReflectionData().sets)
	{
		descriptor_set_layouts.push_back(GraphicsContext::instance()->getDescriptorCache().getDescriptorLayout(pass.shader, set));
		pass_native.descriptor_sets.push_back(DescriptorSet(pass.shader, set));
	}

	// Create pipeline layout
	auto push_constants = pass.shader.getPushConstantRanges();

	VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
	pipeline_layout_create_info.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeline_layout_create_info.pushConstantRangeCount     = static_cast<uint32_t>(push_constants.size());
	pipeline_layout_create_info.pPushConstantRanges        = push_constants.data();
	pipeline_layout_create_info.setLayoutCount             = static_cast<uint32_t>(descriptor_set_layouts.size());
	pipeline_layout_create_info.pSetLayouts                = descriptor_set_layouts.data();

	vkCreatePipelineLayout(GraphicsContext::instance()->getLogicalDevice(), &pipeline_layout_create_info, nullptr, &pass_native.pipeline_layout);

	if (pass.shader.getBindPoint() == VK_PIPELINE_BIND_POINT_GRAPHICS)
	{
		createGraphicsPipeline(pass, pass_native);
	}
	else if (pass.shader.getBindPoint() == VK_PIPELINE_BIND_POINT_COMPUTE)
	{
		createComputePipeline(pass, pass_native);
	}

	return pass_native;
}

std::vector<std::string> RenderGraphBuilder::getRenderPassAttachmentNames(const std::string &render_pass_name, const PipelineMap &pipeline_states)
{
	std::vector<std::string> attachment_names;

	for (const auto &attachment : pipeline_states.at(render_pass_name).getOutputAttachments())
	{
		attachment_names.push_back(attachment.name);
	}

	return attachment_names;
}

RenderGraphBuilder::PipelineBarrierCallback RenderGraphBuilder::createPipelineBarrierCallback(const std::string &render_pass_name, const PipelineState &pipeline_state, const ResourceTransitions &resource_transitions)
{
	auto &buffer_transitions = resource_transitions.buffers.transitions.at(render_pass_name);
	auto &image_transitions  = resource_transitions.images.transitions.at(render_pass_name);

	return [buffer_transitions, image_transitions](const CommandBuffer &command_buffer, const ResolveInfo &resolve_info) {
		insertPipelineBarrier(command_buffer, resolve_info, buffer_transitions, image_transitions);
	};
}
RenderGraphBuilder::CreateCallback RenderGraphBuilder::createOnCreateCallback(const PipelineMap &pipeline_states, const ResourceTransitions &resource_transitions, const AttachmentMap &attachments)
{
	ResolveInfo resolve_info;

	std::unordered_map<std::string, ImageTransition> attachment_transitions;
	for (const auto &[render_pass_name, pipeline_state] : pipeline_states)
	{
		for (const auto &attachment : pipeline_state.getOutputAttachments())
		{
			auto &attachment_transition = resource_transitions.images.transitions.at(render_pass_name).at(attachment.name);
			if (resource_transitions.images.first_usages.at(attachment.name) == render_pass_name)
			{
				attachment_transitions[attachment.name] = ImageTransition{
				    VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM,
				    attachment_transition.initial_usage};
				resolve_info.resolve(attachment.name, attachments.at(attachment.name));
			}
		}
	}

	return [resolve = std::move(resolve_info), transitions = std::move(attachment_transitions)](const CommandBuffer &command_buffer) {
		insertPipelineBarrier(command_buffer, resolve, {}, transitions);
	};
}

RenderGraphBuilder::PresentCallback RenderGraphBuilder::createOnPresentCallback(const std::string &output, const ResourceTransitions &resource_transitions)
{
	// Get output image transition
	const auto &first_render_pass_name = resource_transitions.images.first_usages.at(output);
	const auto &last_render_pass_name  = resource_transitions.images.last_usages.at(output);

	ImageTransition output_image_transition;
	output_image_transition.initial_usage = resource_transitions.images.transitions.at(last_render_pass_name).at(output).final_usage;
	output_image_transition.final_usage   = resource_transitions.images.transitions.at(first_render_pass_name).at(output).initial_usage;

	return [output_image_transition](const CommandBuffer &command_buffer, const Image &output_image, const Image &present_image) {
		command_buffer.blitImage(output_image, output_image_transition.initial_usage, present_image, VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM, VK_FILTER_LINEAR);
		auto subresource_range = output_image.getSubresourceRange();
		if (output_image_transition.final_usage != VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
		{
			VkImageMemoryBarrier barrier = {};
			barrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.srcAccessMask        = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask        = Image::usage_to_access(output_image_transition.final_usage);
			barrier.oldLayout            = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout            = Image::usage_to_layout(output_image_transition.final_usage);
			barrier.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
			barrier.image                = output_image.getImage();
			barrier.subresourceRange     = subresource_range;
			vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, Image::usage_to_stage(output_image_transition.final_usage), 0, 0, nullptr, 0, nullptr, 1, &barrier);
		}

		VkImageMemoryBarrier barrier = {};
		barrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.srcAccessMask        = Image::usage_to_access(VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		barrier.dstAccessMask        = VK_ACCESS_MEMORY_READ_BIT;
		barrier.oldLayout            = Image::usage_to_layout(VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		barrier.newLayout            = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		barrier.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
		barrier.image                = present_image.getImage();
		barrier.subresourceRange     = subresource_range;
		vkCmdPipelineBarrier(command_buffer, Image::usage_to_stage(VK_IMAGE_USAGE_TRANSFER_DST_BIT), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
	};
}
}        // namespace Ilum