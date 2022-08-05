#pragma once

#include "RHI/RHIDevice.hpp"

namespace Ilum
{
class RHIShader;

struct DepthStencilState
{
	bool         depth_test_enable;
	bool         depth_write_enable;
	RHICompareOp compare;
	// TODO: Stencil Test
};

struct BlendState
{
	bool       enable;
	RHILogicOp logic_op;

	struct AttachmentState
	{
		bool           blend_enable;
		RHIBlendFactor src_color_blend;
		RHIBlendFactor dst_color_blend;
		RHIBlendOp     color_blend_op;
		RHIBlendFactor src_alpha_blend;
		RHIBlendFactor dst_alpha_blend;
		RHIBlendOp     alpha_blend_op;
		uint8_t        color_write_mask;
	};

	std::vector<AttachmentState> attachment_states;
};

struct RasterizationState
{
	RHICullMode    cull_mode;
	RHIFrontFace   front_face;
	RHIPolygonMode polygon_mode;
	float          depth_bias;
	float          depth_bias_clamp;
	float          depth_bias_slope;
};

struct MultisampleState
{
	bool     enable;
	uint32_t samples;
	uint32_t sample_mask;
};

struct InputAssemblyState
{
	struct InputAttributeDesc
	{
		uint32_t           location;
		uint32_t           binding;
		RHIFormat          format;
		uint32_t           offset;
		RHIVertexInputRate rate;
	};

	std::vector<InputAttributeDesc> input_attributes;
};

// TODO: Tessellation

class RHIPipelineState
{
  public:
	RHIPipelineState(RHIDevice *device);

	virtual ~RHIPipelineState() = 0;

	RHIPipelineState &SetShader(RHIShaderStage stage, RHIShader *shader);

	RHIPipelineState &SetDepthStencilState(bool depth_test = true, bool depth_write = true, RHICompareOp compare = RHICompareOp::Less_Or_Equal);

	RHIPipelineState &SetBlendState(bool enable = false, RHILogicOp logic_op = RHILogicOp::And);

	RHIPipelineState &AddAttachmentState(
	    bool           enable           = false,
	    RHIBlendFactor src_color        = RHIBlendFactor::Src_Alpha,
	    RHIBlendFactor dst_color        = RHIBlendFactor::One_Minus_Src_Alpha,
	    RHIBlendOp     color_op         = RHIBlendOp::Add,
	    RHIBlendFactor src_alpha        = RHIBlendFactor::One,
	    RHIBlendFactor dst_alpha        = RHIBlendFactor::Zero,
	    RHIBlendOp     alpha_op         = RHIBlendOp::Add,
	    uint8_t        color_write_mask = 1 | 2 | 4 | 8);

	RHIPipelineState &SetRasterizationState(
	    RHICullMode    cull_mode        = RHICullMode::Back,
	    RHIFrontFace   front_face       = RHIFrontFace::Counter_Clockwise,
	    RHIPolygonMode polygon_mode     = RHIPolygonMode::Solid,
	    float          depth_bias       = 0.f,
	    float          depth_bias_clamp = 0.f,
	    float          depth_bias_slope = 0.f);

	RHIPipelineState &SetMultisampleState(
	    bool     enable      = false,
	    uint32_t samples     = 1,
	    uint32_t sample_mask = 0);

	RHIPipelineState &AddInputAttribute(
	    uint32_t           location,
	    uint32_t           binding,
	    RHIFormat          format,
	    uint32_t           offset,
	    RHIVertexInputRate rate);

	void Build();

  private:
	RHIDevice *p_device = nullptr;

	std::unordered_map<RHIShaderStage, RHIShader *> m_shaders;

	DepthStencilState  m_depth_stencil_state;
	BlendState         m_blend_state;
	RasterizationState m_rasterization_state;
	MultisampleState   m_multisample_state;
	InputAssemblyState m_input_assembly_state;

	size_t m_hash = 0;
};
}        // namespace Ilum