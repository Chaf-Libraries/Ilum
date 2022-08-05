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
		m_shaders[stage] = shader;
	}
	else
	{
		if (m_shaders.find(stage) != m_shaders.end())
		{
			m_shaders.erase(stage);
		}
	}

	return *this;
}

RHIPipelineState &RHIPipelineState::SetDepthStencilState(bool depth_test, bool depth_write, RHICompareOp compare)
{
	m_depth_stencil_state.depth_test_enable  = depth_test;
	m_depth_stencil_state.depth_write_enable = depth_write;
	m_depth_stencil_state.compare            = compare;
	return *this;
}

RHIPipelineState &RHIPipelineState::SetBlendState(bool enable, RHILogicOp logic_op)
{
	m_blend_state.enable   = enable;
	m_blend_state.logic_op = logic_op;
	return *this;
}

RHIPipelineState &RHIPipelineState::AddAttachmentState(bool enable, RHIBlendFactor src_color, RHIBlendFactor dst_color, RHIBlendOp color_op, RHIBlendFactor src_alpha, RHIBlendFactor dst_alpha, RHIBlendOp alpha_op, uint8_t color_write_mask)
{
	m_blend_state.attachment_states.push_back(
	    BlendState::AttachmentState{
	        enable,
	        src_color,
	        dst_color,
	        color_op,
	        src_alpha,
	        dst_alpha,
	        alpha_op,
	        color_write_mask});
	return *this;
}

RHIPipelineState &RHIPipelineState::SetRasterizationState(RHICullMode cull_mode, RHIFrontFace front_face, RHIPolygonMode polygon_mode, float depth_bias, float depth_bias_clamp, float depth_bias_slope)
{
	m_rasterization_state.cull_mode        = cull_mode;
	m_rasterization_state.front_face       = front_face;
	m_rasterization_state.polygon_mode     = polygon_mode;
	m_rasterization_state.depth_bias       = depth_bias;
	m_rasterization_state.depth_bias_clamp = depth_bias_clamp;
	m_rasterization_state.depth_bias_slope = depth_bias_slope;
	return *this;
}

RHIPipelineState &RHIPipelineState::SetMultisampleState(bool enable, uint32_t samples, uint32_t sample_mask)
{
	m_multisample_state.enable      = enable;
	m_multisample_state.samples     = samples;
	m_multisample_state.sample_mask = sample_mask;
	return *this;
}

RHIPipelineState &RHIPipelineState::AddInputAttribute(uint32_t location, uint32_t binding, RHIFormat format, uint32_t offset, RHIVertexInputRate rate)
{
	m_input_assembly_state.input_attributes.push_back(
	    InputAssemblyState::InputAttributeDesc{
	        location,
	        binding,
	        format,
	        offset,
	        rate});
	return *this;
}

void RHIPipelineState::Build()
{
	m_hash = 0;

	// Hash Shader
	for (auto &[stage, shader] : m_shaders)
	{
		HashCombine(m_hash, stage, (uint64_t) shader);
	}

	// Hash Depth Stencil State
	HashCombine(
	    m_hash,
	    m_depth_stencil_state.compare,
	    m_depth_stencil_state.depth_test_enable,
	    m_depth_stencil_state.depth_write_enable);

	// Hash Blend State
	HashCombine(
	    m_hash,
	    m_blend_state.enable,
	    m_blend_state.logic_op);

	for (auto &attachment : m_blend_state.attachment_states)
	{
		HashCombine(
		    m_hash,
		    attachment.blend_enable,
		    attachment.src_color_blend,
		    attachment.dst_color_blend,
		    attachment.color_blend_op,
		    attachment.src_alpha_blend,
		    attachment.dst_alpha_blend,
		    attachment.alpha_blend_op,
		    attachment.color_write_mask);
	}

	// Hash Rasterization State
	HashCombine(
	    m_hash,
	    m_rasterization_state.cull_mode,
	    m_rasterization_state.front_face,
	    m_rasterization_state.polygon_mode,
	    m_rasterization_state.depth_bias,
	    m_rasterization_state.depth_bias_clamp,
	    m_rasterization_state.depth_bias_slope);

	// Hash Multisample State
	HashCombine(
	    m_hash,
	    m_multisample_state.enable,
	    m_multisample_state.samples,
	    m_multisample_state.sample_mask);

	// Hash Input Assembly State
	for (auto &input_attribute : m_input_assembly_state.input_attributes)
	{
		HashCombine(
		    m_hash,
		    input_attribute.location,
		    input_attribute.binding,
		    input_attribute.format,
		    input_attribute.offset,
		    input_attribute.rate);
	}
}
}        // namespace Ilum