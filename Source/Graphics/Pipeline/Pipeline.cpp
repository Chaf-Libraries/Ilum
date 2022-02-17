#include "Pipeline.hpp"
#include "Device/Device.hpp"
#include "PipelineLayout.hpp"
#include "PipelineState.hpp"
#include "RenderPass/RenderPass.hpp"
#include "Shader/Shader.hpp"

#include <array>

namespace Ilum::Graphics
{
Pipeline::Pipeline(const Device &device, const PipelineState &pso, const PipelineLayout &pipeline_layout, const RenderPass &render_pass, VkPipelineCache pipeline_cache, uint32_t subpass_index) :
    m_device(device)
{
	// Create graphics pipeline
	assert(pso.GetBindPoint() == VK_PIPELINE_BIND_POINT_GRAPHICS);

	// Input Assembly State
	VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info = {};
	input_assembly_state_create_info.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	input_assembly_state_create_info.topology                               = pso.GetInputAssemblyState().topology;
	input_assembly_state_create_info.flags                                  = 0;
	input_assembly_state_create_info.primitiveRestartEnable                 = pso.GetInputAssemblyState().primitive_restart_enable;

	// Rasterization State
	VkPipelineRasterizationStateCreateInfo rasterization_state_create_info = {};
	rasterization_state_create_info.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterization_state_create_info.depthClampEnable                       = pso.GetRasterizationState().depth_clamp_enable;
	rasterization_state_create_info.rasterizerDiscardEnable                = pso.GetRasterizationState().rasterizer_discard_enable;
	rasterization_state_create_info.polygonMode                            = pso.GetRasterizationState().polygon_mode;
	rasterization_state_create_info.cullMode                               = pso.GetRasterizationState().cull_mode;
	rasterization_state_create_info.frontFace                              = pso.GetRasterizationState().front_face;
	rasterization_state_create_info.depthBiasEnable                        = pso.GetRasterizationState().depth_bias_enable;
	rasterization_state_create_info.lineWidth                              = pso.GetRasterizationState().depth_clamp_enable;

	// Color Blend State
	std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachment_states(pso.GetColorBlendState().attachments.size());
	for (uint32_t i = 0; i < pso.GetColorBlendState().attachments.size(); i++)
	{
		color_blend_attachment_states[i].blendEnable         = pso.GetColorBlendState().attachments[i].blend_enable;
		color_blend_attachment_states[i].srcColorBlendFactor = pso.GetColorBlendState().attachments[i].src_color_blend_factor;
		color_blend_attachment_states[i].dstColorBlendFactor = pso.GetColorBlendState().attachments[i].dst_color_blend_factor;
		color_blend_attachment_states[i].colorBlendOp        = pso.GetColorBlendState().attachments[i].color_blend_op;
		color_blend_attachment_states[i].srcAlphaBlendFactor = pso.GetColorBlendState().attachments[i].src_alpha_blend_factor;
		color_blend_attachment_states[i].dstAlphaBlendFactor = pso.GetColorBlendState().attachments[i].dst_alpha_blend_factor;
		color_blend_attachment_states[i].alphaBlendOp        = pso.GetColorBlendState().attachments[i].alpha_blend_op;
		color_blend_attachment_states[i].colorWriteMask      = pso.GetColorBlendState().attachments[i].color_write_mask;
	}
	VkPipelineColorBlendStateCreateInfo color_blend_state_create_info = {};
	color_blend_state_create_info.sType                               = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blend_state_create_info.logicOpEnable                       = pso.GetColorBlendState().logic_op_enable;
	color_blend_state_create_info.logicOp                             = pso.GetColorBlendState().logic_op;
	color_blend_state_create_info.attachmentCount                     = static_cast<uint32_t>(color_blend_attachment_states.size());
	color_blend_state_create_info.pAttachments                        = color_blend_attachment_states.data();
	color_blend_state_create_info.blendConstants[0]                   = 1.f;
	color_blend_state_create_info.blendConstants[1]                   = 1.f;
	color_blend_state_create_info.blendConstants[2]                   = 1.f;
	color_blend_state_create_info.blendConstants[3]                   = 1.f;

	// Depth Stencil State
	VkPipelineDepthStencilStateCreateInfo depth_stencil_state_create_info = {};
	depth_stencil_state_create_info.sType                                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth_stencil_state_create_info.depthTestEnable                       = pso.GetDepthStencilState().depth_test_enable;
	depth_stencil_state_create_info.depthWriteEnable                      = pso.GetDepthStencilState().depth_write_enable;
	depth_stencil_state_create_info.depthCompareOp                        = pso.GetDepthStencilState().depth_compare_op;
	depth_stencil_state_create_info.depthBoundsTestEnable                 = pso.GetDepthStencilState().depth_bounds_test_enable;
	depth_stencil_state_create_info.stencilTestEnable                     = pso.GetDepthStencilState().stencil_test_enable;
	depth_stencil_state_create_info.front.failOp                          = pso.GetDepthStencilState().front.fail_op;
	depth_stencil_state_create_info.front.passOp                          = pso.GetDepthStencilState().front.pass_op;
	depth_stencil_state_create_info.front.depthFailOp                     = pso.GetDepthStencilState().front.depth_fail_op;
	depth_stencil_state_create_info.front.compareOp                       = pso.GetDepthStencilState().front.compare_op;
	depth_stencil_state_create_info.front.compareMask                     = ~0U;
	depth_stencil_state_create_info.front.writeMask                       = ~0U;
	depth_stencil_state_create_info.front.reference                       = ~0U;
	depth_stencil_state_create_info.back.failOp                           = pso.GetDepthStencilState().back.fail_op;
	depth_stencil_state_create_info.back.passOp                           = pso.GetDepthStencilState().back.pass_op;
	depth_stencil_state_create_info.back.depthFailOp                      = pso.GetDepthStencilState().back.depth_fail_op;
	depth_stencil_state_create_info.back.compareOp                        = pso.GetDepthStencilState().back.compare_op;
	depth_stencil_state_create_info.back.compareMask                      = ~0U;
	depth_stencil_state_create_info.back.writeMask                        = ~0U;
	depth_stencil_state_create_info.back.reference                        = ~0U;

	// Viewport State
	VkPipelineViewportStateCreateInfo viewport_state_create_info = {};
	viewport_state_create_info.sType                             = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport_state_create_info.viewportCount                     = pso.GetViewportState().viewport_count;
	viewport_state_create_info.scissorCount                      = pso.GetViewportState().scissor_count;

	// Multisample State
	VkPipelineMultisampleStateCreateInfo multisample_state_create_info = {};
	multisample_state_create_info.sType                                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample_state_create_info.rasterizationSamples                 = pso.GetMultisampleState().rasterization_samples;
	multisample_state_create_info.sampleShadingEnable                  = pso.GetMultisampleState().sample_shading_enable;
	multisample_state_create_info.minSampleShading                     = pso.GetMultisampleState().min_sample_shading;
	multisample_state_create_info.alphaToCoverageEnable                = pso.GetMultisampleState().alpha_to_coverage_enable;
	multisample_state_create_info.alphaToOneEnable                     = pso.GetMultisampleState().alpha_to_one_enable;

	// Dynamic State
	std::array<VkDynamicState, 9> dynamic_states{
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR,
	    VK_DYNAMIC_STATE_LINE_WIDTH,
	    VK_DYNAMIC_STATE_DEPTH_BIAS,
	    VK_DYNAMIC_STATE_BLEND_CONSTANTS,
	    VK_DYNAMIC_STATE_DEPTH_BOUNDS,
	    VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
	    VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
	    VK_DYNAMIC_STATE_STENCIL_REFERENCE,
	};
	VkPipelineDynamicStateCreateInfo dynamic_state_create_info = {};
	dynamic_state_create_info.sType                            = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic_state_create_info.pDynamicStates                   = dynamic_states.data();
	dynamic_state_create_info.dynamicStateCount                = static_cast<uint32_t>(dynamic_states.size());

	// Vertex Input State
	VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info = {};
	vertex_input_state_create_info.sType                                = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertex_input_state_create_info.vertexAttributeDescriptionCount      = static_cast<uint32_t>(pso.GetVertexInputState().attributes.size());
	vertex_input_state_create_info.pVertexAttributeDescriptions         = pso.GetVertexInputState().attributes.data();
	vertex_input_state_create_info.vertexBindingDescriptionCount        = static_cast<uint32_t>(pso.GetVertexInputState().bindings.size());
	vertex_input_state_create_info.pVertexBindingDescriptions           = pso.GetVertexInputState().bindings.data();

	// Shader stage state
	std::vector<VkPipelineShaderStageCreateInfo> shader_module_create_infos;
	for (auto &shader_stage : pso.GetStageState().shader_stage_states)
	{
		// Specilization constant
		std::vector<uint8_t>                  data;
		std::vector<VkSpecializationMapEntry> map_entries;

		const auto specialization_constant_state = shader_stage.specialization_constants;

		for (const auto specialization_constant : specialization_constant_state)
		{
			map_entries.push_back({specialization_constant.first, static_cast<uint32_t>(data.size()), specialization_constant.second.size()});
			data.insert(data.end(), specialization_constant.second.begin(), specialization_constant.second.end());
		}

		VkSpecializationInfo specialization_info = {};
		specialization_info.mapEntryCount        = static_cast<uint32_t>(map_entries.size());
		specialization_info.pMapEntries          = map_entries.data();
		specialization_info.dataSize             = data.size();
		specialization_info.pData                = data.data();

		VkPipelineShaderStageCreateInfo shader_stage_create_info = {};
		shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shader_stage_create_info.stage                           = shader_stage.shader->GetStage();
		shader_stage_create_info.pName                           = "main";
		shader_stage_create_info.module                          = shader_stage.shader->GetHandle();
		shader_stage_create_info.pSpecializationInfo             = &specialization_info;
		shader_module_create_infos.push_back(shader_stage_create_info);
	}

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

	graphics_pipeline_create_info.layout             = pipeline_layout;
	graphics_pipeline_create_info.renderPass         = render_pass;
	graphics_pipeline_create_info.subpass            = subpass_index;
	graphics_pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
	graphics_pipeline_create_info.basePipelineIndex  = -1;

	vkCreateGraphicsPipelines(m_device, pipeline_cache, 1, &graphics_pipeline_create_info, nullptr, &m_handle);
}

Pipeline::Pipeline(const Device &device, const PipelineState &pso, const PipelineLayout &pipeline_layout, VkPipelineCache pipeline_cache) :
    m_device(device)
{
	assert(pso.GetBindPoint() == VK_SHADER_STAGE_COMPUTE_BIT);

	VkPipelineShaderStageCreateInfo shader_stage_create_info = {};
	shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shader_stage_create_info.stage                           = pso.GetStageState().shader_stage_states[0].shader->GetStage();
	shader_stage_create_info.pName                           = "main";
	shader_stage_create_info.module                          = pso.GetStageState().shader_stage_states[0].shader->GetHandle();

	VkComputePipelineCreateInfo compute_pipeline_create_info = {};
	compute_pipeline_create_info.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	compute_pipeline_create_info.stage                       = shader_stage_create_info;
	compute_pipeline_create_info.layout                      = pipeline_layout;
	compute_pipeline_create_info.basePipelineIndex           = 0;
	compute_pipeline_create_info.basePipelineHandle          = VK_NULL_HANDLE;

	vkCreateComputePipelines(m_device, pipeline_cache, 1, &compute_pipeline_create_info, nullptr, &m_handle);
}

Pipeline::~Pipeline()
{
	if (m_handle)
	{
		vkDestroyPipeline(m_device, m_handle, nullptr);
	}
}

Pipeline::operator const VkPipeline &() const
{
	return m_handle;
}

const VkPipeline &Pipeline::GetHandle() const
{
	return m_handle;
}
}        // namespace Ilum::Graphics