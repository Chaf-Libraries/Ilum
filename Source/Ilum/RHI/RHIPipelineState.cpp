#include "RHIPipelineState.hpp"

namespace Ilum
{
RHIPipelineState::RHIPipelineState(RHIDevice *device) :
    p_device(device)
{
}

RHIPipelineState &RHIPipelineState::SetShader(RHIShaderStage stage, RHIShader *shader)
{
	if (shader)
	{
		if (m_shaders.find(stage) == m_shaders.end() || m_shaders[stage] != shader)
		{
			m_shaders[stage] = shader;
			m_dirty          = true;
		}
	}
	else
	{
		if (m_shaders.find(stage) != m_shaders.end())
		{
			m_shaders.erase(stage);
			m_dirty = true;
		}
	}

	return *this;
}

RHIPipelineState &RHIPipelineState::SetDepthStencilState(const DepthStencilState &state)
{
	if (m_depth_stencil_state != state)
	{
		m_depth_stencil_state = state;
		m_dirty               = true;
	}
	return *this;
}

RHIPipelineState &RHIPipelineState::SetBlendState(const BlendState &state)
{
	if (m_blend_state != state)
	{
		m_blend_state = state;
		m_dirty       = true;
	}
	return *this;
}

RHIPipelineState &RHIPipelineState::SetRasterizationState(const RasterizationState &state)
{
	if (m_rasterization_state != state)
	{
		m_rasterization_state = state;
		m_dirty               = true;
	}
	return *this;
}

RHIPipelineState &RHIPipelineState::SetMultisampleState(const MultisampleState &state)
{
	if (m_multisample_state != state)
	{
		m_multisample_state = state;
		m_dirty             = true;
	}
	return *this;
}

RHIPipelineState &RHIPipelineState::SetInputAssemblyState(const InputAssemblyState &state)
{
	if (m_input_assembly_state != state)
	{
		m_input_assembly_state = state;
		m_dirty                = true;
	}
	return *this;
}

const DepthStencilState &RHIPipelineState::GetDepthStencilState() const
{
	return m_depth_stencil_state;
}

const BlendState &RHIPipelineState::GetBlendState() const
{
	return m_blend_state;
}

const RasterizationState &RHIPipelineState::GetRasterizationState() const
{
	return m_rasterization_state;
}

const MultisampleState &RHIPipelineState::GetMultisampleState() const
{
	return m_multisample_state;
}

const InputAssemblyState &RHIPipelineState::GetInputAssemblyState() const
{
	return m_input_assembly_state;
}

size_t RHIPipelineState::GetHash()
{
	if (m_dirty)
	{
		m_hash = 0;

		// Hash Shader
		for (auto &[stage, shader] : m_shaders)
		{
			HashCombine(m_hash, stage, shader);
		}

		// Hash Depth Stencil State
		HashCombine(m_hash,
		            m_depth_stencil_state.compare,
		            m_depth_stencil_state.depth_test_enable,
		            m_depth_stencil_state.depth_write_enable);

		// Hash Blend State
		HashCombine(m_hash, m_blend_state.enable, m_blend_state.logic_op);
		for (auto &attachment_state : m_blend_state.attachment_states)
		{
			HashCombine(m_hash,
			            attachment_state.blend_enable,
			            attachment_state.src_color_blend,
			            attachment_state.dst_color_blend,
			            attachment_state.color_blend_op,
			            attachment_state.color_write_mask,
			            attachment_state.src_alpha_blend,
			            attachment_state.dst_alpha_blend,
			            attachment_state.alpha_blend_op);
		}

		// Hash Rasterization State
		HashCombine(m_hash,
		            m_rasterization_state.cull_mode,
		            m_rasterization_state.front_face,
		            m_rasterization_state.polygon_mode,
		            m_rasterization_state.depth_bias,
		            m_rasterization_state.depth_bias_clamp,
		            m_rasterization_state.depth_bias_slope);

		// Hash Multisample State
		HashCombine(m_hash,
		            m_multisample_state.enable,
		            m_multisample_state.samples,
		            m_multisample_state.sample_mask);

		// Hash Input Assembly State
		for (auto& input_attribute : m_input_assembly_state.input_attributes)
		{
			HashCombine(m_hash,
			            input_attribute.location,
			            input_attribute.binding,
			            input_attribute.format,
			            input_attribute.offset,
			            input_attribute.rate);
		}

		m_dirty = false;
	}
	return m_hash;
}
}        // namespace Ilum