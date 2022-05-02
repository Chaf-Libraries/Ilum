#include "PipelineState.hpp"
#include "Device.hpp"

namespace Ilum
{
PipelineState::PipelineState(RHIDevice *device):
    p_device(device)
{
}

PipelineState &PipelineState::SetInputAssemblyState(const InputAssemblyState &input_assembly_state)
{
	m_input_assembly_state = input_assembly_state;
	m_dirty                = true;
	return *this;
}

PipelineState &PipelineState::SetRasterizationState(const RasterizationState &rasterization_state)
{
	m_rasterization_state = rasterization_state;
	m_dirty               = true;
	return *this;
}

PipelineState &PipelineState::SetDepthStencilState(const DepthStencilState &depth_stencil_state)
{
	m_depth_stencil_state = depth_stencil_state;
	m_dirty               = true;
	return *this;
}

PipelineState &PipelineState::SetViewportState(const ViewportState &viewport_state)
{
	m_viewport_state = viewport_state;
	m_dirty          = true;
	return *this;
}

PipelineState &PipelineState::SetMultisampleState(const MultisampleState &multisample_state)
{
	m_multisample_state = multisample_state;
	m_dirty             = true;
	return *this;
}

PipelineState &PipelineState::SetDynamicState(const DynamicState &dynamic_state)
{
	m_dynamic_state = dynamic_state;
	m_dirty         = true;
	return *this;
}

PipelineState &PipelineState::SetVertexInputState(const VertexInputState &vertex_input_state)
{
	m_vertex_input_state = vertex_input_state;
	m_dirty              = true;
	return *this;
}

PipelineState &PipelineState::SetColorBlendState(const ColorBlendState &color_blend_state)
{
	m_color_blend_state = color_blend_state;
	m_dirty             = true;
	return *this;
}

PipelineState &PipelineState::LoadShader(const ShaderDesc &desc)
{
	VkShaderModule shader = p_device->LoadShader(desc);
	m_shaders[desc.stage].push_back(std::make_pair(desc.entry_point, shader));
	m_shader_meta += p_device->ReflectShader(shader);
	m_dirty = true;
	return *this;
}

size_t PipelineState::Hash()
{
	if (!m_dirty)
	{
		return m_hash;
	}

	m_hash = 0;
	HashCombine(m_hash, m_input_assembly_state.topology);
	HashCombine(m_hash, m_input_assembly_state.primitive_restart_enable);

	HashCombine(m_hash, m_rasterization_state.polygon_mode);
	HashCombine(m_hash, m_rasterization_state.cull_mode);
	HashCombine(m_hash, m_rasterization_state.front_face);

	HashCombine(m_hash, m_depth_stencil_state.depth_test_enable);
	HashCombine(m_hash, m_depth_stencil_state.depth_write_enable);
	HashCombine(m_hash, m_depth_stencil_state.stencil_test_enable);
	HashCombine(m_hash, m_depth_stencil_state.depth_compare_op);
	HashCombine(m_hash, m_depth_stencil_state.front.compareMask);
	HashCombine(m_hash, m_depth_stencil_state.front.compareOp);
	HashCombine(m_hash, m_depth_stencil_state.front.depthFailOp);
	HashCombine(m_hash, m_depth_stencil_state.front.failOp);
	HashCombine(m_hash, m_depth_stencil_state.front.passOp);
	HashCombine(m_hash, m_depth_stencil_state.front.reference);
	HashCombine(m_hash, m_depth_stencil_state.front.writeMask);
	HashCombine(m_hash, m_depth_stencil_state.back.compareMask);
	HashCombine(m_hash, m_depth_stencil_state.back.compareOp);
	HashCombine(m_hash, m_depth_stencil_state.back.depthFailOp);
	HashCombine(m_hash, m_depth_stencil_state.back.failOp);
	HashCombine(m_hash, m_depth_stencil_state.back.passOp);
	HashCombine(m_hash, m_depth_stencil_state.back.reference);
	HashCombine(m_hash, m_depth_stencil_state.back.writeMask);

	HashCombine(m_hash, m_viewport_state.scissor_count);
	HashCombine(m_hash, m_viewport_state.viewport_count);

	HashCombine(m_hash, m_multisample_state.sample_count);

	for (auto &state : m_dynamic_state.dynamic_states)
	{
		HashCombine(m_hash, state);
	}

	for (auto &description : m_vertex_input_state.binding_descriptions)
	{
		HashCombine(m_hash, description.binding);
		HashCombine(m_hash, description.inputRate);
		HashCombine(m_hash, description.stride);
	}

	for (auto &description : m_vertex_input_state.attribute_descriptions)
	{
		HashCombine(m_hash, description.binding);
		HashCombine(m_hash, description.format);
		HashCombine(m_hash, description.location);
		HashCombine(m_hash, description.offset);
	}

	for (auto& state : m_color_blend_state.attachment_states)
	{
		HashCombine(m_hash, state.blend_enable);
		HashCombine(m_hash, state.src_color_blend_factor);
		HashCombine(m_hash, state.dst_color_blend_factor);
		HashCombine(m_hash, state.color_blend_op);
		HashCombine(m_hash, state.src_alpha_blend_factor);
		HashCombine(m_hash, state.dst_alpha_blend_factor);
		HashCombine(m_hash, state.alpha_blend_op);
		HashCombine(m_hash, state.color_write_mask);
	}

	HashCombine(m_hash, m_shader_meta.Hash());

	return m_hash;
}

}        // namespace Ilum