#include "PipelineState.hpp"
#include "Shader/Shader.hpp"

#include <Core/Hash.hpp>

namespace std
{
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
struct hash<Ilum::Graphics::VertexInputState>
{
	size_t operator()(const Ilum::Graphics::VertexInputState &vertex_input_state) const
	{
		size_t seed = 0;
		for (auto &binding : vertex_input_state.bindings)
		{
			Ilum::Core::HashCombine(seed, binding);
		}
		for (auto &attribute : vertex_input_state.attributes)
		{
			Ilum::Core::HashCombine(seed, attribute);
		}
		return seed;
	}
};

template <>
struct hash<Ilum::Graphics::InputAssemblyState>
{
	size_t operator()(const Ilum::Graphics::InputAssemblyState &input_assembly_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, input_assembly_state.primitive_restart_enable);
		Ilum::Core::HashCombine(seed, static_cast<size_t>(input_assembly_state.topology));
		return seed;
	}
};

template <>
struct hash<Ilum::Graphics::RasterizationState>
{
	size_t operator()(const Ilum::Graphics::RasterizationState &rasterization_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, rasterization_state.depth_clamp_enable);
		Ilum::Core::HashCombine(seed, rasterization_state.rasterizer_discard_enable);
		Ilum::Core::HashCombine(seed, static_cast<size_t>(rasterization_state.polygon_mode));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(rasterization_state.cull_mode));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(rasterization_state.front_face));
		Ilum::Core::HashCombine(seed, rasterization_state.depth_bias_enable);
		return seed;
	}
};

template <>
struct hash<Ilum::Graphics::ViewportState>
{
	size_t operator()(const Ilum::Graphics::ViewportState &viewport_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, viewport_state.viewport_count);
		Ilum::Core::HashCombine(seed, viewport_state.scissor_count);
		return seed;
	}
};

template <>
struct hash<Ilum::Graphics::MultisampleState>
{
	size_t operator()(const Ilum::Graphics::MultisampleState &multisample_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, static_cast<size_t>(multisample_state.rasterization_samples));
		Ilum::Core::HashCombine(seed, multisample_state.sample_shading_enable);
		Ilum::Core::HashCombine(seed, multisample_state.min_sample_shading);
		Ilum::Core::HashCombine(seed, multisample_state.sample_mask);
		Ilum::Core::HashCombine(seed, multisample_state.alpha_to_coverage_enable);
		Ilum::Core::HashCombine(seed, multisample_state.alpha_to_one_enable);
		return seed;
	}
};

template <>
struct hash<Ilum::Graphics::StencilOpState>
{
	size_t operator()(const Ilum::Graphics::StencilOpState &stencil_op_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, static_cast<size_t>(stencil_op_state.fail_op));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(stencil_op_state.pass_op));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(stencil_op_state.depth_fail_op));
		Ilum::Core::HashCombine(seed, static_cast<size_t>(stencil_op_state.compare_op));
		return seed;
	}
};

template <>
struct hash<Ilum::Graphics::DepthStencilState>
{
	size_t operator()(const Ilum::Graphics::DepthStencilState &depth_stencil_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, depth_stencil_state.depth_test_enable);
		Ilum::Core::HashCombine(seed, depth_stencil_state.depth_write_enable);
		Ilum::Core::HashCombine(seed, static_cast<size_t>(depth_stencil_state.depth_compare_op));
		Ilum::Core::HashCombine(seed, depth_stencil_state.depth_bounds_test_enable);
		Ilum::Core::HashCombine(seed, depth_stencil_state.stencil_test_enable);
		Ilum::Core::HashCombine(seed, depth_stencil_state.front);
		Ilum::Core::HashCombine(seed, depth_stencil_state.back);
		return seed;
	}
};

template <>
struct hash<Ilum::Graphics::ColorBlendAttachmentState>
{
	size_t operator()(const Ilum::Graphics::ColorBlendAttachmentState &color_blend_attachment_state) const
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
struct hash<Ilum::Graphics::ColorBlendState>
{
	size_t operator()(const Ilum::Graphics::ColorBlendState &color_blend_state) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, color_blend_state.logic_op_enable);
		Ilum::Core::HashCombine(seed, static_cast<size_t>(color_blend_state.logic_op));
		for (auto &attachment : color_blend_state.attachments)
		{
			Ilum::Core::HashCombine(seed, attachment);
		}
		return seed;
	}
};

template <>
struct hash<Ilum::Graphics::StageState>
{
	size_t operator()(const Ilum::Graphics::StageState &stage_state) const
	{
		size_t seed = 0;
		for (auto &shader_stage : stage_state.shader_stage_states)
		{
			Ilum::Core::HashCombine(seed, (size_t) shader_stage.shader);
			for (auto &[index, constant] : shader_stage.specialization_constants)
			{
				Ilum::Core::HashCombine(seed, index);
				for (auto &n : constant)
				{
					Ilum::Core::HashCombine(seed, n);
				}
			}
		}
		return seed;
	}
};
}        // namespace std

namespace Ilum::Graphics
{
void PipelineState::SetVertexInputState(const VertexInputState &vertex_input_state)
{
	m_vertex_input_state = vertex_input_state;
	UpdateHash();
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

void PipelineState::SetDepthStencilState(const DepthStencilState &depth_stencil_state)
{
	m_depth_stencil_state = depth_stencil_state;
	UpdateHash();
}

void PipelineState::SetColorBlendState(const ColorBlendState &color_blend_state)
{
	m_color_blend_state = color_blend_state;
	UpdateHash();
}

void PipelineState::SetStageState(const StageState &stage_state)
{
	m_stage_state = stage_state;
	UpdateHash();

	// Update reflection data
	m_reflection_data = ReflectionData{};
	for (auto& shader_stage : m_stage_state.shader_stage_states)
	{
		m_reflection_data += shader_stage.shader->GetReflectionData();
	}
}

void PipelineState::SetSubpassIndex(uint32_t subpass_index)
{
	m_subpass_index = subpass_index;
	UpdateHash();
}

const VertexInputState &PipelineState::GetVertexInputState() const
{
	return m_vertex_input_state;
}

const InputAssemblyState &PipelineState::GetInputAssemblyState() const
{
	return m_input_assembly_state;
}

const RasterizationState &PipelineState::GetRasterizationState() const
{
	return m_rasterization_state;
}

const ViewportState &PipelineState::GetViewportState() const
{
	return m_viewport_state;
}

const MultisampleState &PipelineState::GetMultisampleState() const
{
	return m_multisample_state;
}

const DepthStencilState &PipelineState::GetDepthStencilState() const
{
	return m_depth_stencil_state;
}

const ColorBlendState &PipelineState::GetColorBlendState() const
{
	return m_color_blend_state;
}

const StageState &PipelineState::GetStageState() const
{
	return m_stage_state;
}

const ReflectionData &PipelineState::GetReflectionData() const
{
	return m_reflection_data;
}

uint32_t PipelineState::GetSubpassIndex() const
{
	return m_subpass_index;
}

size_t PipelineState::GetHash() const
{
	return m_hash;
}

VkPipelineBindPoint PipelineState::GetBindPoint() const
{
	for (auto &shader_stage_state : m_stage_state.shader_stage_states)
	{
		if (shader_stage_state.shader->GetStage() == VK_SHADER_STAGE_VERTEX_BIT)
		{
			return VK_PIPELINE_BIND_POINT_GRAPHICS;
		}
		else if (shader_stage_state.shader->GetStage() == VK_SHADER_STAGE_COMPUTE_BIT)
		{
			return VK_PIPELINE_BIND_POINT_COMPUTE;
		}
		else if (shader_stage_state.shader->GetStage() == VK_SHADER_STAGE_RAYGEN_BIT_KHR)
		{
			return VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
		}
	}
	return VK_PIPELINE_BIND_POINT_GRAPHICS;
}

void PipelineState::UpdateHash()
{
	m_hash = 0;
	Ilum::Core::HashCombine(m_hash, m_vertex_input_state);
	Ilum::Core::HashCombine(m_hash, m_input_assembly_state);
	Ilum::Core::HashCombine(m_hash, m_rasterization_state);
	Ilum::Core::HashCombine(m_hash, m_viewport_state);
	Ilum::Core::HashCombine(m_hash, m_multisample_state);
	Ilum::Core::HashCombine(m_hash, m_depth_stencil_state);
	Ilum::Core::HashCombine(m_hash, m_color_blend_state);
	Ilum::Core::HashCombine(m_hash, m_stage_state);
	Ilum::Core::HashCombine(m_hash, m_subpass_index);
}
}        // namespace Ilum::Graphics