#include "RHIPipelineState.hpp"
#include "RHIDevice.hpp"

#include <Core/Hash.hpp>
#include <Core/Plugin.hpp>

namespace Ilum
{
RHIPipelineState::RHIPipelineState(RHIDevice *device) :
    p_device(device)
{
}

std::unique_ptr<RHIPipelineState> RHIPipelineState::Create(RHIDevice *device)
{
	return std::unique_ptr<RHIPipelineState>(std::move(PluginManager::GetInstance().Call<RHIPipelineState *>(fmt::format("shared/RHI/RHI.{}.dll", device->GetBackend()), "CreatePipelineState", device)));
}

RHIPipelineState &RHIPipelineState::SetShader(RHIShaderStage stage, RHIShader *shader)
{
	m_shaders.emplace_back(std::make_pair(stage, shader));
	m_dirty = true;
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

RHIPipelineState &RHIPipelineState::SetVertexInputState(const VertexInputState &state)
{
	if (m_vertex_input_state != state)
	{
		m_vertex_input_state = state;
		m_dirty              = true;
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

RHIPipelineState& RHIPipelineState::ClearShader()
{
	m_shaders.clear();
	return *this;
}

const std::vector<std::pair<RHIShaderStage, RHIShader *>> &RHIPipelineState::GetShaders() const
{
	return m_shaders;
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

const VertexInputState &RHIPipelineState::GetVertexInputState() const
{
	return m_vertex_input_state;
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
		            m_rasterization_state.depth_bias_enable,
		            m_rasterization_state.depth_clamp_enable,
		            m_rasterization_state.depth_bias,
		            m_rasterization_state.depth_bias_clamp,
		            m_rasterization_state.depth_bias_slope);

		// Hash Multisample State
		HashCombine(m_hash,
		            m_multisample_state.enable,
		            m_multisample_state.samples,
		            m_multisample_state.sample_mask);

		// Hash Vertex Input State
		for (auto &input_attribute : m_vertex_input_state.input_attributes)
		{
			HashCombine(m_hash,
			            input_attribute.location,
			            input_attribute.binding,
			            input_attribute.format,
			            input_attribute.offset);
		}

		// Hash Input Assembly State
		HashCombine(m_hash, m_input_assembly_state.topology);

		m_dirty = false;
	}
	return m_hash;
}
}        // namespace Ilum