#include "PipelineState.hpp"
#include "Definitions.hpp"
#include "Descriptor.hpp"
#include "Device.hpp"
#include "RenderTarget.hpp"
#include "Shader.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
static VkPipelineCache                              PipelineCache;
static std::unordered_map<size_t, VkPipeline>       Pipelines;
static std::unordered_map<size_t, VkPipelineLayout> PipelineLayouts;

static uint32_t PipelineCount = 0;

PipelineState::PipelineState(RHIDevice *device) :
    RHIPipelineState(device)
{
	if (PipelineCount++ == 0)
	{
		if (!PipelineCache)
		{
			VkPipelineCacheCreateInfo create_info = {};
			create_info.sType                     = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
			vkCreatePipelineCache(static_cast<Device *>(p_device)->GetDevice(), &create_info, nullptr, &PipelineCache);
		}
	}
}

PipelineState ::~PipelineState()
{
	if (--PipelineCount == 0)
	{
		for (auto &[hash, pipeline] : Pipelines)
		{
			vkDestroyPipeline(static_cast<Device *>(p_device)->GetDevice(), pipeline, nullptr);
		}
		for (auto &[hash, layout] : PipelineLayouts)
		{
			vkDestroyPipelineLayout(static_cast<Device *>(p_device)->GetDevice(), layout, nullptr);
		}
		Pipelines.clear();
		PipelineLayouts.clear();

		if (PipelineCache)
		{
			vkDestroyPipelineCache(static_cast<Device *>(p_device)->GetDevice(), PipelineCache, nullptr);
			PipelineCache = VK_NULL_HANDLE;
		}
	}
}

VkPipelineLayout PipelineState::GetPipelineLayout(Descriptor *descriptor)
{
	size_t hash = 0;
	HashCombine(hash, descriptor->GetShaderMeta().hash, GetHash());

	if (PipelineLayouts.find(hash) != PipelineLayouts.end())
	{
		return PipelineLayouts[hash];
	}

	return CreatePipelineLayout(descriptor);
}

VkPipeline PipelineState::GetPipeline(Descriptor *descriptor, RenderTarget *render_target)
{
	size_t hash = 0;
	HashCombine(hash, descriptor->GetShaderMeta().hash, GetHash());

	if (render_target)
	{
		HashCombine(hash, render_target->GetRenderPass());
	}

	if (Pipelines.find(hash) != Pipelines.end())
	{
		return Pipelines[hash];
	}

	if (m_shaders.find(RHIShaderStage::Fragment) != m_shaders.end())
	{
		ASSERT(render_target != nullptr);
		return CreateGraphicsPipeline(descriptor, render_target);
	}
	else if (m_shaders.find(RHIShaderStage::Compute) != m_shaders.end())
	{
		return CreateComputePipeline(descriptor);
	}
	else if (m_shaders.find(RHIShaderStage::RayGen) != m_shaders.end())
	{
		return CreateRayTracingPipeline(descriptor);
	}
	return VK_NULL_HANDLE;
}

VkPipelineBindPoint PipelineState::GetPipelineBindPoint() const
{
	if (m_shaders.find(RHIShaderStage::Fragment) != m_shaders.end())
	{
		return VK_PIPELINE_BIND_POINT_GRAPHICS;
	}
	else if (m_shaders.find(RHIShaderStage::Compute) != m_shaders.end())
	{
		return VK_PIPELINE_BIND_POINT_COMPUTE;
	}
	else if (m_shaders.find(RHIShaderStage::RayGen) != m_shaders.end())
	{
		return VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
	}
	return VK_PIPELINE_BIND_POINT_GRAPHICS;
}

VkPipelineLayout PipelineState::CreatePipelineLayout(Descriptor *descriptor)
{
	size_t hash = 0;
	HashCombine(hash, descriptor->GetShaderMeta().hash, GetHash());

	std::vector<VkPushConstantRange> push_constants;
	for (auto &constant : descriptor->GetShaderMeta().constants)
	{
		VkPushConstantRange push_constant_range = {};
		push_constant_range.stageFlags          = ToVulkanShaderStages(constant.stage);
		push_constant_range.size                = constant.size;
		push_constant_range.offset              = constant.offset;
		push_constants.push_back(push_constant_range);
	}

	std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
	for (auto &[set, layout] : descriptor->GetDescriptorSetLayout())
	{
		descriptor_set_layouts.push_back(layout);
	}

	VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
	pipeline_layout_create_info.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeline_layout_create_info.pushConstantRangeCount     = static_cast<uint32_t>(push_constants.size());
	pipeline_layout_create_info.pPushConstantRanges        = push_constants.data();
	pipeline_layout_create_info.setLayoutCount             = static_cast<uint32_t>(descriptor_set_layouts.size());
	pipeline_layout_create_info.pSetLayouts                = descriptor_set_layouts.data();

	VkPipelineLayout layout = VK_NULL_HANDLE;
	vkCreatePipelineLayout(static_cast<Device *>(p_device)->GetDevice(), &pipeline_layout_create_info, nullptr, &layout);

	PipelineLayouts.emplace(hash, layout);

	return layout;
}

VkPipeline PipelineState::CreateGraphicsPipeline(Descriptor *descriptor, RenderTarget *render_target)
{
	size_t hash = 0;
	HashCombine(hash, descriptor->GetShaderMeta().hash, GetHash());

	if (render_target)
	{
		HashCombine(hash, render_target->GetRenderPass());
	}

	// Input Assembly State
	VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info = {};
	input_assembly_state_create_info.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	input_assembly_state_create_info.topology                               = ToVulkanPrimitiveTopology[m_input_assembly_state.topology];
	input_assembly_state_create_info.flags                                  = 0;
	input_assembly_state_create_info.primitiveRestartEnable                 = VK_FALSE;

	// Rasterization State
	VkPipelineRasterizationStateCreateInfo rasterization_state_create_info = {};
	rasterization_state_create_info.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterization_state_create_info.polygonMode                            = ToVulkanPolygonMode[m_rasterization_state.polygon_mode];
	rasterization_state_create_info.cullMode                               = ToVulkanCullMode[m_rasterization_state.cull_mode];
	rasterization_state_create_info.frontFace                              = ToVulkanFrontFace[m_rasterization_state.front_face];
	rasterization_state_create_info.flags                                  = 0;
	rasterization_state_create_info.depthBiasEnable                        = VK_TRUE;
	rasterization_state_create_info.lineWidth                              = 1;

	// Color Blend State
	std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachment_states(m_blend_state.attachment_states.size());

	for (uint32_t i = 0; i < color_blend_attachment_states.size(); i++)
	{
		color_blend_attachment_states[i].blendEnable         = m_blend_state.attachment_states[i].blend_enable;
		color_blend_attachment_states[i].srcColorBlendFactor = ToVulkanBlendFactor[m_blend_state.attachment_states[i].src_color_blend];
		color_blend_attachment_states[i].dstColorBlendFactor = ToVulkanBlendFactor[m_blend_state.attachment_states[i].dst_color_blend];
		color_blend_attachment_states[i].colorBlendOp        = ToVulkanBlendOp[m_blend_state.attachment_states[i].color_blend_op];
		color_blend_attachment_states[i].srcAlphaBlendFactor = ToVulkanBlendFactor[m_blend_state.attachment_states[i].src_alpha_blend];
		color_blend_attachment_states[i].dstAlphaBlendFactor = ToVulkanBlendFactor[m_blend_state.attachment_states[i].dst_alpha_blend];
		color_blend_attachment_states[i].alphaBlendOp        = ToVulkanBlendOp[m_blend_state.attachment_states[i].alpha_blend_op];
		color_blend_attachment_states[i].colorWriteMask      = m_blend_state.attachment_states[i].color_write_mask;
	}

	VkPipelineColorBlendStateCreateInfo color_blend_state_create_info = {};
	color_blend_state_create_info.sType                               = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blend_state_create_info.logicOpEnable                       = m_blend_state.enable;
	color_blend_state_create_info.logicOp                             = ToVulkanLogicOp[m_blend_state.logic_op];
	color_blend_state_create_info.attachmentCount                     = static_cast<uint32_t>(color_blend_attachment_states.size());
	color_blend_state_create_info.pAttachments                        = color_blend_attachment_states.data();
	color_blend_state_create_info.blendConstants[0]                   = m_blend_state.blend_constants[0];
	color_blend_state_create_info.blendConstants[1]                   = m_blend_state.blend_constants[1];
	color_blend_state_create_info.blendConstants[2]                   = m_blend_state.blend_constants[2];
	color_blend_state_create_info.blendConstants[3]                   = m_blend_state.blend_constants[3];

	// Depth Stencil State
	VkPipelineDepthStencilStateCreateInfo depth_stencil_state_create_info = {};
	depth_stencil_state_create_info.sType                                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth_stencil_state_create_info.depthTestEnable                       = m_depth_stencil_state.depth_test_enable;
	depth_stencil_state_create_info.depthWriteEnable                      = m_depth_stencil_state.depth_write_enable;
	depth_stencil_state_create_info.depthCompareOp                        = ToVulkanCompareOp[m_depth_stencil_state.compare];
	// TODO: stencil test
	depth_stencil_state_create_info.back              = VkStencilOpState{};
	depth_stencil_state_create_info.front             = VkStencilOpState{};
	depth_stencil_state_create_info.stencilTestEnable = VK_FALSE;

	// Viewport State
	VkPipelineViewportStateCreateInfo viewport_state_create_info = {};
	viewport_state_create_info.sType                             = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport_state_create_info.viewportCount                     = 1;
	viewport_state_create_info.scissorCount                      = 1;
	viewport_state_create_info.flags                             = 0;

	// Multisample State
	VkPipelineMultisampleStateCreateInfo multisample_state_create_info = {};
	multisample_state_create_info.sType                                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample_state_create_info.rasterizationSamples                 = ToVulkanSampleCount[m_multisample_state.samples];
	multisample_state_create_info.sampleShadingEnable                  = m_multisample_state.enable;
	multisample_state_create_info.pSampleMask                          = &m_multisample_state.sample_mask;
	multisample_state_create_info.flags                                = 0;

	// Dynamic State
	std::vector<VkDynamicState>      dynamic_states            = {VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_VIEWPORT};
	VkPipelineDynamicStateCreateInfo dynamic_state_create_info = {};
	dynamic_state_create_info.sType                            = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic_state_create_info.pDynamicStates                   = dynamic_states.data();
	dynamic_state_create_info.dynamicStateCount                = static_cast<uint32_t>(dynamic_states.size());
	dynamic_state_create_info.flags                            = 0;

	// Shader Stage State
	std::vector<VkPipelineShaderStageCreateInfo> pipeline_shader_stage_create_infos;
	for (auto &[stage, shader] : m_shaders)
	{
		VkPipelineShaderStageCreateInfo shader_stage_create_info = {};

		shader_stage_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shader_stage_create_info.stage  = ToVulkanShaderStage[stage];
		shader_stage_create_info.module = static_cast<Shader *>(shader)->GetHandle();
		shader_stage_create_info.pName  = shader->GetEntryPoint().c_str();
		pipeline_shader_stage_create_infos.push_back(shader_stage_create_info);
	}

	// Vertex Input State
	std::vector<VkVertexInputAttributeDescription> attribute_descriptions = {};
	std::vector<VkVertexInputBindingDescription>   binding_descriptions   = {};

	for (auto &attribute : m_vertex_input_state.input_attributes)
	{
		attribute_descriptions.push_back(VkVertexInputAttributeDescription{
		    attribute.location,
		    attribute.binding,
		    ToVulkanFormat[attribute.format],
		    attribute.offset});
	}

	for (auto &binding : m_vertex_input_state.input_bindings)
	{
		binding_descriptions.push_back(VkVertexInputBindingDescription{
		    binding.binding,
		    binding.stride,
		    ToVulkanVertexInputRate[binding.rate]});
	}

	VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info = {};
	vertex_input_state_create_info.sType                                = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertex_input_state_create_info.vertexAttributeDescriptionCount      = static_cast<uint32_t>(attribute_descriptions.size());
	vertex_input_state_create_info.pVertexAttributeDescriptions         = attribute_descriptions.data();
	vertex_input_state_create_info.vertexBindingDescriptionCount        = static_cast<uint32_t>(binding_descriptions.size());
	vertex_input_state_create_info.pVertexBindingDescriptions           = binding_descriptions.data();

	VkGraphicsPipelineCreateInfo graphics_pipeline_create_info = {};
	graphics_pipeline_create_info.sType                        = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	graphics_pipeline_create_info.stageCount                   = static_cast<uint32_t>(pipeline_shader_stage_create_infos.size());
	graphics_pipeline_create_info.pStages                      = pipeline_shader_stage_create_infos.data();

	graphics_pipeline_create_info.pInputAssemblyState = &input_assembly_state_create_info;
	graphics_pipeline_create_info.pRasterizationState = &rasterization_state_create_info;
	graphics_pipeline_create_info.pColorBlendState    = &color_blend_state_create_info;
	graphics_pipeline_create_info.pDepthStencilState  = &depth_stencil_state_create_info;
	graphics_pipeline_create_info.pViewportState      = &viewport_state_create_info;
	graphics_pipeline_create_info.pMultisampleState   = &multisample_state_create_info;
	graphics_pipeline_create_info.pDynamicState       = &dynamic_state_create_info;
	graphics_pipeline_create_info.pVertexInputState   = &vertex_input_state_create_info;

	graphics_pipeline_create_info.layout             = GetPipelineLayout(descriptor);
	graphics_pipeline_create_info.renderPass         = render_target->GetRenderPass();
	graphics_pipeline_create_info.subpass            = 0;
	graphics_pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
	graphics_pipeline_create_info.basePipelineIndex  = -1;

	//VkPipelineRenderingCreateInfo pipeline_rendering_create_info = {};
	//pipeline_rendering_create_info.sType                         = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
	//pipeline_rendering_create_info.colorAttachmentCount          = static_cast<uint32_t>(render_target->GetColorFormat().size());
	//pipeline_rendering_create_info.pColorAttachmentFormats       = render_target->GetColorFormat().data();
	//pipeline_rendering_create_info.depthAttachmentFormat         = render_target->GetDepthFormat().has_value() ? render_target->GetDepthFormat().value() : VK_FORMAT_UNDEFINED;
	//pipeline_rendering_create_info.stencilAttachmentFormat       = render_target->GetStencilFormat().has_value() ? render_target->GetStencilFormat().value() : VK_FORMAT_UNDEFINED;

	//graphics_pipeline_create_info.pNext = &pipeline_rendering_create_info;

	VkPipeline pipeline = VK_NULL_HANDLE;
	vkCreateGraphicsPipelines(static_cast<Device *>(p_device)->GetDevice(), PipelineCache, 1, &graphics_pipeline_create_info, nullptr, &pipeline);

	Pipelines.emplace(hash, pipeline);

	return pipeline;
}

VkPipeline PipelineState::CreateComputePipeline(Descriptor *descriptor)
{
	size_t hash = 0;
	HashCombine(hash, descriptor->GetShaderMeta().hash, GetHash());

	VkPipelineShaderStageCreateInfo shader_stage_create_info = {};
	shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shader_stage_create_info.stage                           = VK_SHADER_STAGE_COMPUTE_BIT;
	shader_stage_create_info.module                          = static_cast<Shader *>(m_shaders.at(RHIShaderStage::Compute))->GetHandle();
	shader_stage_create_info.pName                           = m_shaders.at(RHIShaderStage::Compute)->GetEntryPoint().c_str();

	VkComputePipelineCreateInfo compute_pipeline_create_info = {};
	compute_pipeline_create_info.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	compute_pipeline_create_info.stage                       = shader_stage_create_info;
	compute_pipeline_create_info.layout                      = GetPipelineLayout(descriptor);
	compute_pipeline_create_info.basePipelineIndex           = 0;
	compute_pipeline_create_info.basePipelineHandle          = VK_NULL_HANDLE;

	VkPipeline pipeline = VK_NULL_HANDLE;
	vkCreateComputePipelines(static_cast<Device *>(p_device)->GetDevice(), PipelineCache, 1, &compute_pipeline_create_info, nullptr, &pipeline);

	Pipelines.emplace(hash, pipeline);

	return pipeline;
}

VkPipeline PipelineState::CreateRayTracingPipeline(Descriptor *descriptor)
{
	return VK_NULL_HANDLE;
}
}        // namespace Ilum::Vulkan