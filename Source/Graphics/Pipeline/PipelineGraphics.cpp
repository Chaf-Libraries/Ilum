#include "PipelineGraphics.hpp"
#include "PipelineState.hpp"

#include "Device/LogicalDevice.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/RenderPass/RenderTarget.hpp"

namespace Ilum
{
PipelineGraphics::PipelineGraphics(const std::vector<std::string> &shader_paths, const RenderTarget &render_target, PipelineState pipeline_state, uint32_t subpass_index, const Shader::Variant &variant) :
    m_pipeline_state(pipeline_state)
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
	std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachment_states(render_target.getSubpassAttachmentCounts()[subpass_index]);
	if (color_blend_attachment_states.size() == 1)
	{
		color_blend_attachment_states[0].blendEnable         = VK_TRUE;
		color_blend_attachment_states[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		color_blend_attachment_states[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		color_blend_attachment_states[0].colorBlendOp        = VK_BLEND_OP_ADD;
		color_blend_attachment_states[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		color_blend_attachment_states[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;
		color_blend_attachment_states[0].alphaBlendOp        = VK_BLEND_OP_MAX;
		color_blend_attachment_states[0].colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	}
	else if (color_blend_attachment_states.size() >= 2)
	{
		for (auto &color_blend_attachment_state : color_blend_attachment_states)
		{
			color_blend_attachment_state.blendEnable         = VK_TRUE;
			color_blend_attachment_state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
			color_blend_attachment_state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			color_blend_attachment_state.colorBlendOp        = VK_BLEND_OP_ADD;
			color_blend_attachment_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
			color_blend_attachment_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			color_blend_attachment_state.alphaBlendOp        = VK_BLEND_OP_ADD;
			color_blend_attachment_state.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		}
	}

	VkPipelineColorBlendStateCreateInfo color_blend_state_create_info = {};
	color_blend_state_create_info.sType                                            = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blend_state_create_info.logicOpEnable                                    = VK_FALSE;
	color_blend_state_create_info.logicOp                                          = VK_LOGIC_OP_COPY;
	color_blend_state_create_info.attachmentCount                                  = static_cast<uint32_t>(color_blend_attachment_states.size());
	color_blend_state_create_info.pAttachments                                     = color_blend_attachment_states.data();
	color_blend_state_create_info.blendConstants[0]                                = 0.0f;
	color_blend_state_create_info.blendConstants[1]                                = 0.0f;
	color_blend_state_create_info.blendConstants[2]                                = 0.0f;
	color_blend_state_create_info.blendConstants[3]                                = 0.0f;

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

	std::vector<VkPipelineShaderStageCreateInfo> shader_module_create_infos(shader_paths.size());
	for (uint32_t i = 0; i < shader_paths.size(); i++)
	{
		auto &shader_stage  = shader_module_create_infos[i];
		shader_stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shader_stage.stage  = Shader::getShaderStage(shader_paths[i]);
		shader_stage.pName  = "main";
		shader_stage.module = m_shader->createShaderModule(shader_paths[i], variant);
	}

	// Vertex Input State
	VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info = {};
	vertex_input_state_create_info.sType                                = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertex_input_state_create_info.vertexAttributeDescriptionCount      = static_cast<uint32_t>(pipeline_state.vertex_input_state.attribute_descriptions.size());
	vertex_input_state_create_info.pVertexAttributeDescriptions         = pipeline_state.vertex_input_state.attribute_descriptions.data();
	vertex_input_state_create_info.vertexBindingDescriptionCount        = static_cast<uint32_t>(pipeline_state.vertex_input_state.binding_descriptions.size());
	vertex_input_state_create_info.pVertexBindingDescriptions           = pipeline_state.vertex_input_state.binding_descriptions.data();

	createPipelineLayout();

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

	graphics_pipeline_create_info.layout             = m_pipeline_layout;
	graphics_pipeline_create_info.renderPass         = render_target.getRenderPass();
	graphics_pipeline_create_info.subpass            = subpass_index;
	graphics_pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
	graphics_pipeline_create_info.basePipelineIndex  = -1;

	vkCreateGraphicsPipelines(GraphicsContext::instance()->getLogicalDevice(), GraphicsContext::instance()->getPipelineCache(), 1, &graphics_pipeline_create_info, nullptr, &m_pipeline);
}
}        // namespace Ilum