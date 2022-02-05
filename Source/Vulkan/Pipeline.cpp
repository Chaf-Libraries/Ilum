#include "Pipeline.hpp"
#include "Descriptor.hpp"
#include "Device.hpp"
#include "RenderContext.hpp"
#include "RenderPass.hpp"
#include "Shader.hpp"

#include "ShaderCompiler/SpirvReflection.hpp"

#include <Core/Hash.hpp>

namespace std
{
template <>
struct hash<Ilum::Vulkan::InputAssemblyState>
{
	size_t operator()(const Ilum::Vulkan::InputAssemblyState &input_assembly_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, input_assembly_state.primitive_restart_enable);
		Ilum::Core::HashCombine(seed, static_cast<size_t>(input_assembly_state.topology));
		return seed;
	}
};

template <>
struct hash<Ilum::Vulkan::RasterizationState>
{
	size_t operator()(const Ilum::Vulkan::RasterizationState &rasterization_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, static_cast<size_t>(rasterization_state.cull_mode));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(rasterization_state.front_face));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(rasterization_state.line_width));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(rasterization_state.polygon_mode));
		Ilum::Core::HashCombine(seed, rasterization_state.depth_bias_clamp);
		Ilum::Core::HashCombine(seed, rasterization_state.depth_bias_constant_factor);
		Ilum::Core::HashCombine(seed, rasterization_state.depth_bias_enable);
		Ilum::Core::HashCombine(seed, rasterization_state.depth_bias_slope_factor);
		Ilum::Core::HashCombine(seed, rasterization_state.depth_clamp_enable);
		Ilum::Core::HashCombine(seed, rasterization_state.rasterizer_discard_enable);
		return seed;
	}
};

template <>
struct hash<Ilum::Vulkan::ColorBlendAttachmentState>
{
	size_t operator()(const Ilum::Vulkan::ColorBlendAttachmentState &color_blend_attachment_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, color_blend_attachment_state.blend_enable);
		Ilum::Core::HashCombine(seed, static_cast<size_t>(color_blend_attachment_state.src_color_blend_factor));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(color_blend_attachment_state.dst_color_blend_factor));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(color_blend_attachment_state.color_blend_op));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(color_blend_attachment_state.src_alpha_blend_factor));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(color_blend_attachment_state.dst_alpha_blend_factor));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(color_blend_attachment_state.alpha_blend_op));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(color_blend_attachment_state.color_write_mask));
		return seed;
	}
};

template <>
struct hash<Ilum::Vulkan::ColorBlendState>
{
	size_t operator()(const Ilum::Vulkan::ColorBlendState &color_blend_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, color_blend_state.logic_op_enable);
		Ilum::Core::HashCombine(seed, color_blend_state.blendConstants[0]);
		Ilum::Core::HashCombine(seed, color_blend_state.blendConstants[1]);
		Ilum::Core::HashCombine(seed, color_blend_state.blendConstants[2]);
		Ilum::Core::HashCombine(seed, color_blend_state.blendConstants[3]);
		Ilum::Core::HashCombine(seed, static_cast<size_t>(color_blend_state.logic_op));
		for (auto &attachment : color_blend_state.color_blend_attachments)
		{
			Ilum::Core::HashCombine(seed, attachment);
		}
		return seed;
	}
};

template <>
struct hash<VkStencilOpState>
{
	size_t operator()(const VkStencilOpState &stencil_op_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, stencil_op_state.compareMask);
		Ilum::Core::HashCombine(seed, stencil_op_state.writeMask);
		Ilum::Core::HashCombine(seed, stencil_op_state.reference);
		Ilum::Core::HashCombine(seed, static_cast<size_t>(stencil_op_state.failOp));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(stencil_op_state.passOp));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(stencil_op_state.depthFailOp));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(stencil_op_state.compareOp));
		return seed;
	}
};

template <>
struct hash<Ilum::Vulkan::DepthStencilState>
{
	size_t operator()(const Ilum::Vulkan::DepthStencilState &depth_stencil_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, depth_stencil_state.depth_test_enable);
		Ilum::Core::HashCombine(seed, depth_stencil_state.depth_write_enable);
		Ilum::Core::HashCombine(seed, static_cast<size_t>(depth_stencil_state.depth_compare_op));
		Ilum::Core::HashCombine(seed, depth_stencil_state.depth_bounds_test_enable);
		Ilum::Core::HashCombine(seed, depth_stencil_state.stencil_test_enable);
		Ilum::Core::HashCombine(seed, depth_stencil_state.front);
		Ilum::Core::HashCombine(seed, depth_stencil_state.back);
		Ilum::Core::HashCombine(seed, depth_stencil_state.min_depth_bounds);
		Ilum::Core::HashCombine(seed, depth_stencil_state.max_depth_bounds);
		return seed;
	}
};

template <>
struct hash<VkViewport>
{
	size_t operator()(const VkViewport &viewport) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, viewport.x);
		Ilum::Core::HashCombine(seed, viewport.y);
		Ilum::Core::HashCombine(seed, viewport.width);
		Ilum::Core::HashCombine(seed, viewport.height);
		Ilum::Core::HashCombine(seed, viewport.minDepth);
		Ilum::Core::HashCombine(seed, viewport.maxDepth);
		return seed;
	}
};

template <>
struct hash<VkRect2D>
{
	size_t operator()(const VkRect2D &scissor) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, scissor.extent.width);
		Ilum::Core::HashCombine(seed, scissor.extent.height);
		Ilum::Core::HashCombine(seed, scissor.offset.x);
		Ilum::Core::HashCombine(seed, scissor.offset.y);
		return seed;
	}
};

template <>
struct hash<Ilum::Vulkan::ViewportState>
{
	size_t operator()(const Ilum::Vulkan::ViewportState &viewport_state) const
	{
		size_t seed = 0;
		for (auto &viewport : viewport_state.viewports)
		{
			Ilum::Core::HashCombine(seed, viewport);
		}
		for (auto &scissor : viewport_state.scissors)
		{
			Ilum::Core::HashCombine(seed, scissor);
		}
		return seed;
	}
};

template <>
struct hash<Ilum::Vulkan::MultisampleState>
{
	size_t operator()(const Ilum::Vulkan::MultisampleState &multisample_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, static_cast<size_t>(multisample_state.rasterization_samples));
		Ilum::Core::HashCombine(seed, multisample_state.sample_shading_enable);
		Ilum::Core::HashCombine(seed, multisample_state.min_sample_shading);
		Ilum::Core::HashCombine(seed, multisample_state.alpha_to_coverage_enable);
		Ilum::Core::HashCombine(seed, multisample_state.alpha_to_one_enable);
		for (auto &sample_mask : multisample_state.sample_mask)
		{
			Ilum::Core::HashCombine(seed, sample_mask);
		}
		return seed;
	}
};

template <>
struct hash<Ilum::Vulkan::DynamicState>
{
	size_t operator()(const Ilum::Vulkan::DynamicState &dynamic_state) const
	{
		size_t seed = 0;
		for (auto &state : dynamic_state.dynamic_states)
		{
			Ilum::Core::HashCombine(seed, static_cast<size_t>(state));
		}
		return seed;
	}
};

template <>
struct hash<VkVertexInputBindingDescription>
{
	size_t operator()(const VkVertexInputBindingDescription &binding_descriptions) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, binding_descriptions.binding);
		Ilum::Core::HashCombine(seed, binding_descriptions.stride);
		Ilum::Core::HashCombine(seed, static_cast<size_t>(binding_descriptions.inputRate));
		return seed;
	}
};

template <>
struct hash<VkVertexInputAttributeDescription>
{
	size_t operator()(const VkVertexInputAttributeDescription &attribute_descriptions) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, attribute_descriptions.binding);
		Ilum::Core::HashCombine(seed, attribute_descriptions.location);
		Ilum::Core::HashCombine(seed, attribute_descriptions.offset);
		Ilum::Core::HashCombine(seed, static_cast<size_t>(attribute_descriptions.format));
		return seed;
	}
};

template <>
struct hash<Ilum::Vulkan::VertexInputState>
{
	size_t operator()(const Ilum::Vulkan::VertexInputState &vertex_input_state) const
	{
		size_t seed = 0;
		for (auto &binding_description : vertex_input_state.binding_descriptions)
		{
			Ilum::Core::HashCombine(seed, binding_description);
		}
		for (auto &attribute_description : vertex_input_state.attribute_descriptions)
		{
			Ilum::Core::HashCombine(seed, attribute_description);
		}
		return seed;
	}
};

template <>
struct hash<Ilum::Vulkan::ShaderState>
{
	size_t operator()(const Ilum::Vulkan::ShaderState &shader_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, (size_t) shader_state.shader);
		for (auto &specialization : shader_state.specializations)
		{
			Ilum::Core::HashCombine(seed, specialization.mapEntryCount);
			Ilum::Core::HashCombine(seed, (size_t) specialization.pMapEntries);
			Ilum::Core::HashCombine(seed, specialization.dataSize);
			Ilum::Core::HashCombine(seed, (size_t) specialization.pData);
		}
		return seed;
	}
};

template <>
struct hash<Ilum::Vulkan::ShaderStageState>
{
	size_t operator()(const Ilum::Vulkan::ShaderStageState &shader_stage_state) const
	{
		size_t seed = 0;
		for (auto &shader : shader_stage_state.shaders)
		{
			Ilum::Core::HashCombine(seed, shader);
		}
		return seed;
	}
};
}        // namespace std

namespace Ilum::Vulkan
{
const InputAssemblyState &PipelineState::GetInputAssemblyState() const
{
	return m_input_assembly_state;
}

const RasterizationState &PipelineState::GetRasterizationState() const
{
	return m_rasterization_state;
}

const ColorBlendState &PipelineState::GetColorBlendState() const
{
	return m_color_blend_state;
}

const DepthStencilState &PipelineState::GetDepthStencilState() const
{
	return m_depth_stencil_state;
}

const ViewportState &PipelineState::GetViewportState() const
{
	return m_viewport_state;
}

const MultisampleState &PipelineState::GetMultisampleState() const
{
	return m_multisample_state;
}

const DynamicState &PipelineState::GetDynamicState() const
{
	return m_dynamic_state;
}

const VertexInputState &PipelineState::GetVertexInputState() const
{
	return m_vertex_input_state;
}

const ShaderStageState &PipelineState::GetShaderStageState() const
{
	return m_shader_stage_state;
}

void PipelineState::SetInputAssemblyState(const InputAssemblyState &input_assembly_state)
{
	m_input_assembly_state = input_assembly_state;
	UpdateHash();
}

void PipelineState::SetRasterizationState(const RasterizationState &rasterization_state)
{
	m_rasterization_state = rasterization_state;
	UpdateHash();
}

void PipelineState::SetColorBlendState(const ColorBlendState &color_blend_state)
{
	m_color_blend_state = color_blend_state;
	UpdateHash();
}

void PipelineState::SetDepthStencilState(const DepthStencilState &depth_stencil_state)
{
	m_depth_stencil_state = depth_stencil_state;
	UpdateHash();
}

void PipelineState::SetViewportState(const ViewportState &viewport_state)
{
	m_viewport_state = viewport_state;
	UpdateHash();
}

void PipelineState::SetMultisampleState(const MultisampleState &multisample_state)
{
	m_multisample_state = multisample_state;
	UpdateHash();
}

void PipelineState::SetDynamicState(const DynamicState &dynamic_state)
{
	m_dynamic_state = dynamic_state;
	UpdateHash();
}

void PipelineState::SetVertexInputState(const VertexInputState &vertex_input_state)
{
	m_vertex_input_state = vertex_input_state;
	UpdateHash();
}

void PipelineState::SetShaderStageState(const ShaderStageState &shader_stage_state)
{
	m_shader_stage_state = shader_stage_state;
	UpdateHash();
}

size_t PipelineState::GetHash() const
{
	return m_hash;
}

VkPipelineBindPoint PipelineState::GetBindPoint() const
{
	for (auto &shader : m_shader_stage_state.shaders)
	{
		if (shader.shader->GetStage() == VK_SHADER_STAGE_VERTEX_BIT)
		{
			return VK_PIPELINE_BIND_POINT_GRAPHICS;
		}
		else if (shader.shader->GetStage() == VK_SHADER_STAGE_COMPUTE_BIT)
		{
			return VK_PIPELINE_BIND_POINT_COMPUTE;
		}
		else if (shader.shader->GetStage() == VK_SHADER_STAGE_RAYGEN_BIT_KHR)
		{
			return VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
		}
	}
	return VK_PIPELINE_BIND_POINT_GRAPHICS;
}

void PipelineState::UpdateHash()
{
	m_hash = 0;
	Ilum::Core::HashCombine(m_hash, m_input_assembly_state);
	Ilum::Core::HashCombine(m_hash, m_rasterization_state);
	Ilum::Core::HashCombine(m_hash, m_color_blend_state);
	Ilum::Core::HashCombine(m_hash, m_depth_stencil_state);
	Ilum::Core::HashCombine(m_hash, m_viewport_state);
	Ilum::Core::HashCombine(m_hash, m_multisample_state);
	Ilum::Core::HashCombine(m_hash, m_dynamic_state);
	Ilum::Core::HashCombine(m_hash, m_vertex_input_state);
	Ilum::Core::HashCombine(m_hash, m_shader_stage_state);
}
PipelineLayout::PipelineLayout(const ReflectionData &reflection_data)
{
	std::vector<VkPushConstantRange> push_constant_ranges;
	for (auto &constant : reflection_data.constants)
	{
		if (constant.type == ReflectionData::Constant::Type::Push)
		{
			VkPushConstantRange push_constant_range = {};
			push_constant_range.stageFlags          = constant.stage;
			push_constant_range.size                = constant.size;
			push_constant_range.offset              = constant.offset;
			push_constant_ranges.push_back(push_constant_range);
		}
	}

	std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
	for (auto &set : reflection_data.sets)
	{
		auto &layout = RenderContext::GetDescriptorCache().RequestDescriptorSetLayout(reflection_data, set);
		descriptor_set_layouts.push_back(layout);
	}

	VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
	pipeline_layout_create_info.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeline_layout_create_info.pushConstantRangeCount     = static_cast<uint32_t>(push_constant_ranges.size());
	pipeline_layout_create_info.pPushConstantRanges        = push_constant_ranges.data();
	pipeline_layout_create_info.setLayoutCount             = static_cast<uint32_t>(descriptor_set_layouts.size());
	pipeline_layout_create_info.pSetLayouts                = descriptor_set_layouts.data();

	vkCreatePipelineLayout(RenderContext::GetDevice(), &pipeline_layout_create_info, nullptr, &m_handle);
}

PipelineLayout::~PipelineLayout()
{
	if (m_handle)
	{
		vkDestroyPipelineLayout(RenderContext::GetDevice(), m_handle, nullptr);
	}
}

PipelineLayout::operator const VkPipelineLayout &() const
{
	return m_handle;
}

const VkPipelineLayout &PipelineLayout::GetHandle() const
{
	return m_handle;
}

void PipelineLayout::SetName(const std::string &name) const
{
	VKDebugger::SetName(m_handle, name.c_str());
}

Pipeline::Pipeline(const PipelineState &pso, const PipelineLayout &pipeline_layout, const RenderPass &render_pass, VkPipelineCache pipeline_cache, uint32_t subpass_index)
{
	// Create graphics pipeline
	assert(pso.GetBindPoint() == VK_PIPELINE_BIND_POINT_GRAPHICS);

	// Input Assembly State
	auto &                                 input_assembly_state             = pso.GetInputAssemblyState();
	VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info = {};
	input_assembly_state_create_info.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	input_assembly_state_create_info.topology                               = input_assembly_state.topology;
	input_assembly_state_create_info.flags                                  = 0;
	input_assembly_state_create_info.primitiveRestartEnable                 = input_assembly_state.primitive_restart_enable;

	// Rasterization State
	auto &                                 rasterization_state             = pso.GetRasterizationState();
	VkPipelineRasterizationStateCreateInfo rasterization_state_create_info = {};
	rasterization_state_create_info.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterization_state_create_info.depthClampEnable                       = rasterization_state.depth_clamp_enable;
	rasterization_state_create_info.rasterizerDiscardEnable                = rasterization_state.rasterizer_discard_enable;
	rasterization_state_create_info.polygonMode                            = rasterization_state.polygon_mode;
	rasterization_state_create_info.cullMode                               = rasterization_state.cull_mode;
	rasterization_state_create_info.frontFace                              = rasterization_state.front_face;
	rasterization_state_create_info.depthBiasEnable                        = rasterization_state.depth_bias_enable;
	rasterization_state_create_info.depthBiasConstantFactor                = rasterization_state.depth_bias_constant_factor;
	rasterization_state_create_info.depthBiasClamp                         = rasterization_state.depth_bias_clamp;
	rasterization_state_create_info.depthBiasSlopeFactor                   = rasterization_state.depth_bias_slope_factor;
	rasterization_state_create_info.lineWidth                              = rasterization_state.depth_clamp_enable;

	// Color Blend State
	auto &                                           color_blend_state = pso.GetColorBlendState();
	std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachment_states(color_blend_state.color_blend_attachments.size());
	for (uint32_t i = 0; i < color_blend_state.color_blend_attachments.size(); i++)
	{
		color_blend_attachment_states[i].blendEnable         = color_blend_state.color_blend_attachments[i].blend_enable;
		color_blend_attachment_states[i].srcColorBlendFactor = color_blend_state.color_blend_attachments[i].src_color_blend_factor;
		color_blend_attachment_states[i].dstColorBlendFactor = color_blend_state.color_blend_attachments[i].dst_color_blend_factor;
		color_blend_attachment_states[i].colorBlendOp        = color_blend_state.color_blend_attachments[i].color_blend_op;
		color_blend_attachment_states[i].srcAlphaBlendFactor = color_blend_state.color_blend_attachments[i].src_alpha_blend_factor;
		color_blend_attachment_states[i].dstAlphaBlendFactor = color_blend_state.color_blend_attachments[i].dst_alpha_blend_factor;
		color_blend_attachment_states[i].alphaBlendOp        = color_blend_state.color_blend_attachments[i].alpha_blend_op;
		color_blend_attachment_states[i].colorWriteMask      = color_blend_state.color_blend_attachments[i].color_write_mask;
	}
	VkPipelineColorBlendStateCreateInfo color_blend_state_create_info = {};
	color_blend_state_create_info.sType                               = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blend_state_create_info.logicOpEnable                       = color_blend_state.logic_op_enable;
	color_blend_state_create_info.logicOp                             = color_blend_state.logic_op;
	color_blend_state_create_info.attachmentCount                     = static_cast<uint32_t>(color_blend_attachment_states.size());
	color_blend_state_create_info.pAttachments                        = color_blend_attachment_states.data();
	color_blend_state_create_info.blendConstants[0]                   = color_blend_state.blendConstants[0];
	color_blend_state_create_info.blendConstants[1]                   = color_blend_state.blendConstants[1];
	color_blend_state_create_info.blendConstants[2]                   = color_blend_state.blendConstants[2];
	color_blend_state_create_info.blendConstants[3]                   = color_blend_state.blendConstants[3];

	// Depth Stencil State
	auto &                                depth_stencil_state             = pso.GetDepthStencilState();
	VkPipelineDepthStencilStateCreateInfo depth_stencil_state_create_info = {};
	depth_stencil_state_create_info.sType                                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth_stencil_state_create_info.depthTestEnable                       = depth_stencil_state.depth_test_enable;
	depth_stencil_state_create_info.depthWriteEnable                      = depth_stencil_state.depth_write_enable;
	depth_stencil_state_create_info.depthCompareOp                        = depth_stencil_state.depth_compare_op;
	depth_stencil_state_create_info.depthBoundsTestEnable                 = depth_stencil_state.depth_bounds_test_enable;
	depth_stencil_state_create_info.stencilTestEnable                     = depth_stencil_state.stencil_test_enable;
	depth_stencil_state_create_info.back                                  = depth_stencil_state.back;
	depth_stencil_state_create_info.front                                 = depth_stencil_state.front;
	depth_stencil_state_create_info.minDepthBounds                        = depth_stencil_state.min_depth_bounds;
	depth_stencil_state_create_info.maxDepthBounds                        = depth_stencil_state.max_depth_bounds;

	// Viewport State
	auto &                            viewport_state             = pso.GetViewportState();
	VkPipelineViewportStateCreateInfo viewport_state_create_info = {};
	viewport_state_create_info.sType                             = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport_state_create_info.viewportCount                     = static_cast<uint32_t>(viewport_state.viewports.size());
	viewport_state_create_info.pViewports                        = viewport_state.viewports.data();
	viewport_state_create_info.scissorCount                      = static_cast<uint32_t>(viewport_state.scissors.size());
	viewport_state_create_info.pScissors                         = viewport_state.scissors.data();

	// Multisample State
	auto &                               multisample_state             = pso.GetMultisampleState();
	VkPipelineMultisampleStateCreateInfo multisample_state_create_info = {};
	multisample_state_create_info.sType                                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample_state_create_info.rasterizationSamples                 = multisample_state.rasterization_samples;
	multisample_state_create_info.sampleShadingEnable                  = multisample_state.sample_shading_enable;
	multisample_state_create_info.minSampleShading                     = multisample_state.min_sample_shading;
	multisample_state_create_info.pSampleMask                          = multisample_state.sample_mask.data();
	multisample_state_create_info.alphaToCoverageEnable                = multisample_state.alpha_to_coverage_enable;
	multisample_state_create_info.alphaToOneEnable                     = multisample_state.alpha_to_one_enable;

	// Dynamic State
	auto &                           dynamic_state             = pso.GetDynamicState();
	VkPipelineDynamicStateCreateInfo dynamic_state_create_info = {};
	dynamic_state_create_info.sType                            = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic_state_create_info.pDynamicStates                   = dynamic_state.dynamic_states.data();
	dynamic_state_create_info.dynamicStateCount                = static_cast<uint32_t>(dynamic_state.dynamic_states.size());

	// Vertex Input State
	auto &                               vertex_input_state             = pso.GetVertexInputState();
	VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info = {};
	vertex_input_state_create_info.sType                                = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertex_input_state_create_info.vertexAttributeDescriptionCount      = static_cast<uint32_t>(vertex_input_state.attribute_descriptions.size());
	vertex_input_state_create_info.pVertexAttributeDescriptions         = vertex_input_state.attribute_descriptions.data();
	vertex_input_state_create_info.vertexBindingDescriptionCount        = static_cast<uint32_t>(vertex_input_state.binding_descriptions.size());
	vertex_input_state_create_info.pVertexBindingDescriptions           = vertex_input_state.binding_descriptions.data();

	// Shader stage state
	auto &                                       shader_stage_state = pso.GetShaderStageState();
	std::vector<VkPipelineShaderStageCreateInfo> shader_module_create_infos;
	for (auto &shader_stage : shader_stage_state.shaders)
	{
		VkPipelineShaderStageCreateInfo shader_stage_create_info = {};

		shader_stage_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shader_stage_create_info.stage  = shader_stage.shader->GetStage();
		shader_stage_create_info.pName  = "main";
		shader_stage_create_info.module = *shader_stage.shader;
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

	vkCreateGraphicsPipelines(RenderContext::GetDevice(), pipeline_cache, 1, &graphics_pipeline_create_info, nullptr, &m_handle);
}

Pipeline::Pipeline(const PipelineState &pso, const PipelineLayout &pipeline_layout, VkPipelineCache pipeline_cache)
{
	assert(pso.GetShaderStageState().shaders[0].shader->GetStage() == VK_SHADER_STAGE_COMPUTE_BIT);

	VkPipelineShaderStageCreateInfo shader_stage_create_info = {};
	shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shader_stage_create_info.stage                           = VK_SHADER_STAGE_COMPUTE_BIT;
	shader_stage_create_info.pName                           = "main";
	shader_stage_create_info.module                          = *pso.GetShaderStageState().shaders[0].shader;

	VkComputePipelineCreateInfo compute_pipeline_create_info = {};
	compute_pipeline_create_info.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	compute_pipeline_create_info.stage                       = shader_stage_create_info;
	compute_pipeline_create_info.layout                      = pipeline_layout;
	compute_pipeline_create_info.basePipelineIndex           = 0;
	compute_pipeline_create_info.basePipelineHandle          = VK_NULL_HANDLE;

	vkCreateComputePipelines(RenderContext::GetDevice(), pipeline_cache, 1, &compute_pipeline_create_info, nullptr, &m_handle);
}

Pipeline::~Pipeline()
{
	if (m_handle)
	{
		vkDestroyPipeline(RenderContext::GetDevice(), m_handle, nullptr);
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

void Pipeline::SetName(const std::string &name) const
{
	VKDebugger::SetName(m_handle, name.c_str());
}

const PipelineLayout &PipelineCache::RequestPipelineLayout(const PipelineState &pso)
{
	if (m_pipeline_layouts.find(pso.GetHash()) != m_pipeline_layouts.end())
	{
		return *m_pipeline_layouts[pso.GetHash()];
	}

	{
		std::lock_guard<std::mutex> lock(m_layout_mutex);
		ReflectionData              reflection_data = {};
		for (auto &shader_stage : pso.GetShaderStageState().shaders)
		{
			reflection_data += shader_stage.shader->GetReflectionData();
		}

		m_pipeline_layouts.emplace(pso.GetHash(), std::make_unique<PipelineLayout>(reflection_data));
	}

	return *m_pipeline_layouts[pso.GetHash()];
}

const Pipeline &PipelineCache::RequestPipeline(const PipelineState &pso, const RenderPass &render_pass, uint32_t subpass_index)
{
	size_t hash = 0;
	Core::HashCombine(hash, pso.GetHash());
	Core::HashCombine(hash, render_pass.GetHash());
	Core::HashCombine(hash, subpass_index);

	if (m_pipelines.find(hash) != m_pipelines.end())
	{
		return *m_pipelines[hash];
	}

	{
		std::lock_guard<std::mutex> lock(m_pipeline_mutex);
		m_pipelines.emplace(hash, std::make_unique<Pipeline>(pso, RequestPipelineLayout(pso), render_pass, m_handle, subpass_index));
	}

	return *m_pipelines[hash];
}

const Pipeline &PipelineCache::RequestPipeline(const PipelineState &pso)
{
	if (m_pipelines.find(pso.GetHash()) != m_pipelines.end())
	{
		return *m_pipelines[pso.GetHash()];
	}

	{
		std::lock_guard<std::mutex> lock(m_pipeline_mutex);
		m_pipelines.emplace(pso.GetHash(), std::make_unique<Pipeline>(pso, RequestPipelineLayout(pso), m_handle));
	}

	return *m_pipelines[pso.GetHash()];
}
}        // namespace Ilum::Vulkan