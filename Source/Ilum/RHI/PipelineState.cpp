#include "PipelineState.hpp"
#include "Device.hpp"

namespace Ilum
{
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
	if (m_bind_point == VK_PIPELINE_BIND_POINT_MAX_ENUM)
	{
		if (desc.stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR)
		{
			m_bind_point = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
		}
		else if (desc.stage == VK_SHADER_STAGE_COMPUTE_BIT)
		{
			m_bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
		}
		else
		{
			m_bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
		}
	}

	m_shaders.push_back(desc);
	m_dirty = true;
	return *this;
}

const InputAssemblyState &PipelineState::GetInputAssemblyState() const
{
	return m_input_assembly_state;
}

const RasterizationState &PipelineState::GetRasterizationState() const
{
	return m_rasterization_state;
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

const ColorBlendState &PipelineState::GetColorBlendState() const
{
	return m_color_blend_state;
}

const std::vector<ShaderDesc> &PipelineState::GetShaders() const
{
	return m_shaders;
}

VkPipelineBindPoint PipelineState::GetBindPoint() const
{
	return m_bind_point;
}

size_t PipelineState::Hash()
{
	if (!m_dirty)
	{
		return m_hash;
	}

	m_hash = 0;
	HashCombine(m_hash, m_input_assembly_state.Hash());
	HashCombine(m_hash, m_rasterization_state.Hash());
	HashCombine(m_hash, m_depth_stencil_state.Hash());
	HashCombine(m_hash, m_viewport_state.Hash());
	HashCombine(m_hash, m_multisample_state.Hash());
	HashCombine(m_hash, m_dynamic_state.Hash());
	HashCombine(m_hash, m_vertex_input_state.Hash());
	HashCombine(m_hash, m_color_blend_state.Hash());
	for (auto& desc : m_shaders)
	{
		HashCombine(m_hash, desc.Hash());
	}

	m_dirty = false;

	return m_hash;
}

}        // namespace Ilum