#pragma once

#include "RHI/RHIDevice.hpp"

namespace Ilum
{
class RHIShader;

struct DepthStencilState
{
	bool         depth_test_enable  = true;
	bool         depth_write_enable = true;
	RHICompareOp compare            = RHICompareOp::Less_Or_Equal;
	// TODO: Stencil Test

	inline bool operator==(const DepthStencilState &state)
	{
		return depth_test_enable == state.depth_test_enable &&
		       depth_write_enable == state.depth_write_enable &&
		       compare == state.compare;
	}
	inline bool operator!=(const DepthStencilState &state)
	{
		return !(*this == state);
	}
};

struct BlendState
{
	bool       enable   = false;
	RHILogicOp logic_op = RHILogicOp::And;

	struct AttachmentState
	{
		bool           blend_enable     = false;
		RHIBlendFactor src_color_blend  = RHIBlendFactor::Src_Alpha;
		RHIBlendFactor dst_color_blend  = RHIBlendFactor::One_Minus_Src_Alpha;
		RHIBlendOp     color_blend_op   = RHIBlendOp::Add;
		RHIBlendFactor src_alpha_blend  = RHIBlendFactor::One;
		RHIBlendFactor dst_alpha_blend  = RHIBlendFactor::Zero;
		RHIBlendOp     alpha_blend_op   = RHIBlendOp::Add;
		uint8_t        color_write_mask = 1 | 2 | 4 | 8;

		inline bool operator==(const AttachmentState &state)
		{
			return blend_enable == state.blend_enable &&
			       src_color_blend == state.src_color_blend &&
			       dst_color_blend == state.dst_color_blend &&
			       color_blend_op == state.color_blend_op &&
			       src_alpha_blend == state.src_alpha_blend &&
			       dst_alpha_blend == state.dst_alpha_blend &&
			       alpha_blend_op == state.alpha_blend_op &&
			       color_write_mask == state.color_write_mask;
		}
		inline bool operator!=(const AttachmentState &state)
		{
			return !(*this == state);
		}
	};

	std::vector<AttachmentState> attachment_states;

	inline bool operator==(const BlendState &state)
	{
		if (attachment_states.size() == state.attachment_states.size())
		{
			for (uint32_t i = 0; i < attachment_states.size(); i++)
			{
				if (attachment_states[i] != state.attachment_states[i])
				{
					return false;
				}
			}
			return true;
		}
		else
		{
			return false;
		}
	}
	inline bool operator!=(const BlendState &state)
	{
		return !(*this == state);
	}
};

struct RasterizationState
{
	RHICullMode    cull_mode        = RHICullMode::Back;
	RHIFrontFace   front_face       = RHIFrontFace::Counter_Clockwise;
	RHIPolygonMode polygon_mode     = RHIPolygonMode::Solid;
	float          depth_bias       = 0.f;
	float          depth_bias_clamp = 0.f;
	float          depth_bias_slope = 0.f;

	inline bool operator==(const RasterizationState &state)
	{
		return cull_mode == state.cull_mode &&
		       front_face == state.front_face &&
		       polygon_mode == state.polygon_mode &&
		       depth_bias == state.depth_bias &&
		       depth_bias_clamp == state.depth_bias_clamp &&
		       depth_bias_slope == state.depth_bias_slope;
	}
	inline bool operator!=(const RasterizationState &state)
	{
		return !(*this == state);
	}
};

struct MultisampleState
{
	bool     enable      = false;
	uint32_t samples     = 1;
	uint32_t sample_mask = 0;

	inline bool operator==(const MultisampleState &state)
	{
		return enable == state.enable &&
		       samples == state.samples &&
		       sample_mask == state.sample_mask;
	}
	inline bool operator!=(const MultisampleState &state)
	{
		return !(*this == state);
	}
};

struct InputAssemblyState
{
	// TODO: optimize
	struct InputAttribute
	{
		uint32_t           location;
		uint32_t           binding;
		RHIFormat          format;
		uint32_t           offset;
		RHIVertexInputRate rate;

		inline bool operator==(const InputAttribute &desc)
		{
			return location == desc.location &&
			       binding == desc.binding &&
			       format == desc.format &&
			       offset == desc.offset &&
			       rate == desc.rate;
		}
		inline bool operator!=(const InputAttribute &state)
		{
			return !(*this == state);
		}
	};

	std::vector<InputAttribute> input_attributes;

	inline bool operator==(const InputAssemblyState &state)
	{
		if (input_attributes.size() == state.input_attributes.size())
		{
			for (uint32_t i = 0; i < input_attributes.size(); i++)
			{
				if (input_attributes[i] != state.input_attributes[i])
				{
					return false;
				}
			}
			return true;
		}
		else
		{
			return false;
		}
	}
	inline bool operator!=(const InputAssemblyState &state)
	{
		return !(*this == state);
	}
};

// TODO: Tessellation

class RHIPipelineState
{
  public:
	RHIPipelineState(RHIDevice *device);

	virtual ~RHIPipelineState() = default;

	RHIPipelineState &SetShader(RHIShaderStage stage, RHIShader *shader);

	RHIPipelineState &SetDepthStencilState(const DepthStencilState &state);
	RHIPipelineState &SetBlendState(const BlendState &state);
	RHIPipelineState &SetRasterizationState(const RasterizationState &state);
	RHIPipelineState &SetMultisampleState(const MultisampleState &state);
	RHIPipelineState &SetInputAssemblyState(const InputAssemblyState &state);

	const DepthStencilState  &GetDepthStencilState() const;
	const BlendState         &GetBlendState() const;
	const RasterizationState &GetRasterizationState() const;
	const MultisampleState   &GetMultisampleState() const;
	const InputAssemblyState &GetInputAssemblyState() const;

	size_t GetHash();

  protected:
	RHIDevice *p_device = nullptr;

	std::unordered_map<RHIShaderStage, RHIShader *> m_shaders;

	DepthStencilState  m_depth_stencil_state;
	BlendState         m_blend_state;
	RasterizationState m_rasterization_state;
	MultisampleState   m_multisample_state;
	InputAssemblyState m_input_assembly_state;

	bool   m_dirty = false;
	size_t m_hash  = 0;
};
}        // namespace Ilum