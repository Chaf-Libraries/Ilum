#include "GraphicsPipeline.hpp"

namespace Ilum
{
GraphicsPipeline::GraphicsPipeline(const LogicalDevice &logical_device, VkPipelineCache pipeline_cache, const std::vector<std::string> &shader_paths, PipelineState pipeline_state, const Shader::Variant &variant) :
    Pipeline(logical_device), m_pipeline_state(pipeline_state)
{
	std::vector<VkShaderModule> m_shader_modules;
	for (auto &shader_path : shader_paths)
	{
		m_shader_modules.push_back(m_shader->createShaderModule(shader_path, variant));
	}

	VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info = {};
	input_assembly_state_create_info.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	input_assembly_state_create_info.topology                               = pipeline_state.input_assembly_state.topology;
	input_assembly_state_create_info.flags                                  = 0;
	input_assembly_state_create_info.primitiveRestartEnable                 = pipeline_state.input_assembly_state.primitive_restart_enable;

	VkPipelineRasterizationStateCreateInfo rasterization_state_create_info = {};
	rasterization_state_create_info.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterization_state_create_info.polygonMode                            = pipeline_state.rasterization_state.polygon_mode;
	rasterization_state_create_info.cullMode                               = pipeline_state.rasterization_state.cull_mode;
	rasterization_state_create_info.frontFace                              = pipeline_state.rasterization_state.front_face;
	rasterization_state_create_info.flags                                  = 0;
	rasterization_state_create_info.depthClampEnable                       = VK_FALSE;
	rasterization_state_create_info.lineWidth                              = 1.0f;

	std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachment_states;
	for (auto &color_blend_attachment_state : pipeline_state.color_blend_state.color_blend_attachment_states)
	{
		VkPipelineColorBlendAttachmentState color_blend_attachment_state_{};
		color_blend_attachment_state_.colorWriteMask = color_blend_attachment_state.color_write_mask;
		color_blend_attachment_state_.blendEnable    = color_blend_attachment_state.blend_enable;
		color_blend_attachment_states.push_back(color_blend_attachment_state_);
	}

	VkPipelineColorBlendStateCreateInfo color_blend_state_create_info{};
	color_blend_state_create_info.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blend_state_create_info.attachmentCount = static_cast<uint32_t>(color_blend_attachment_states.size());
	color_blend_state_create_info.pAttachments    = color_blend_attachment_states.data();

	VkPipelineDepthStencilStateCreateInfo depth_stencil_state_create_info{};
	depth_stencil_state_create_info.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth_stencil_state_create_info.depthTestEnable  = pipeline_state.depth_stencil_state.depth_test_enable;
	depth_stencil_state_create_info.depthWriteEnable = pipeline_state.depth_stencil_state.depth_write_enable;
	depth_stencil_state_create_info.depthCompareOp   = pipeline_state.depth_stencil_state.depth_compare_op;
	depth_stencil_state_create_info.back.compareOp   = VK_COMPARE_OP_ALWAYS;

	VkPipelineViewportStateCreateInfo viewport_state_create_info{};
	viewport_state_create_info.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport_state_create_info.viewportCount = pipeline_state.viewport_state.viewport_count;
	viewport_state_create_info.scissorCount  = pipeline_state.viewport_state.scissor_count;
	viewport_state_create_info.flags         = 0;

	VkPipelineMultisampleStateCreateInfo multisample_state_create_info{};
	multisample_state_create_info.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample_state_create_info.rasterizationSamples = pipeline_state.multisample_state.sample_count;
	multisample_state_create_info.flags                = 0;

	VkPipelineDynamicStateCreateInfo dynamic_state_create_info{};
	dynamic_state_create_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic_state_create_info.pDynamicStates         = pipeline_state.dynamic_state.dynamic_states.data();
	dynamic_state_create_info.dynamicStateCount      = static_cast<uint32_t>(pipeline_state.dynamic_state.dynamic_states.size());
	dynamic_state_create_info.flags                  = 0;

	std::vector<VkPipelineShaderStageCreateInfo> shader_module_create_infos(shader_paths.size());
	for (uint32_t i = 0; i < shader_paths.size(); i++)
	{
		auto& shader_stage = shader_module_create_infos[i];
		shader_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shader_stage.stage = Shader::getShaderStage(shader_paths[i]);
		shader_stage.pName = "main";
		shader_stage.module = m_shader->createShaderModule(shader_paths[i], variant);
	}

	// TODO: RenderPass, PipelineLayout -> Create Pipeline
}

GraphicsPipeline::~GraphicsPipeline()
{
}
}        // namespace Ilum