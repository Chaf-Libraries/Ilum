#include "PipelineState.hpp"
#include "Definitions.hpp"
#include "Descriptor.hpp"
#include "Device.hpp"
#include "Shader.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
static VkPipelineCache                              PipelineCache;
static std::unordered_map<size_t, VkPipeline>       Pipelines;
static std::unordered_map<size_t, VkPipelineLayout> PipelineLayouts;

static uint32_t PipelineCount = 0;

PipelineState::PipelineState(RHIDevice *device, Descriptor *descriptor) :
    RHIPipelineState(device), m_meta(descriptor->GetShaderMeta())
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

VkPipelineLayout PipelineState::GetPipelineLayout() const
{
	size_t hash = 0;
	HashCombine(hash, m_meta.hash, m_hash);

	if (PipelineLayouts.find(hash) != PipelineLayouts.end())
	{
		return PipelineLayouts[hash];
	}

	return CreatePipelineLayout();
}

VkPipeline PipelineState::GetPipeline() const
{
	return CreatePipeline();
}

VkPipelineLayout PipelineState::CreatePipelineLayout() const
{
	size_t hash = 0;
	HashCombine(hash, m_meta.hash, m_hash);

	std::vector<VkPushConstantRange> push_constants;
	for (auto &constant : m_meta.constants)
	{
		VkPushConstantRange push_constant_range = {};
		push_constant_range.stageFlags          = ToVulkanShaderStages(constant.stage);
		push_constant_range.size                = constant.size;
		push_constant_range.offset              = constant.offset;
		push_constants.push_back(push_constant_range);
	}

	std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
	for (auto &[set, layout] : m_descriptor->GetDescriptorSetLayout())
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

VkPipeline PipelineState::CreatePipeline() const
{
	size_t hash = 0;
	HashCombine(hash, m_meta.hash, m_hash);

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
}
}        // namespace Ilum::Vulkan