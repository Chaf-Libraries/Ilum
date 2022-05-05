#pragma once

#include "ShaderCompiler.hpp"
#include "ShaderReflection.hpp"
#include "Texture.hpp"

#include <map>
#include <string>
#include <vector>

namespace Ilum
{
class RHIDevice;

struct InputAssemblyState
{
	VkPrimitiveTopology topology                 = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	bool                primitive_restart_enable = false;

	size_t Hash() const
	{
		size_t hash = 0;
		HashCombine(hash, topology);
		HashCombine(hash, primitive_restart_enable);

		return hash;
	}
};

struct RasterizationState
{
	VkPolygonMode   polygon_mode = VK_POLYGON_MODE_FILL;
	VkCullModeFlags cull_mode    = VK_CULL_MODE_BACK_BIT;
	VkFrontFace     front_face   = VK_FRONT_FACE_CLOCKWISE;

	size_t Hash() const
	{
		size_t hash = 0;
		HashCombine(hash, polygon_mode);
		HashCombine(hash, cull_mode);
		HashCombine(hash, front_face);

		return hash;
	}
};

struct DepthStencilState
{
	bool             depth_test_enable   = true;
	bool             depth_write_enable  = true;
	bool             stencil_test_enable = true;
	VkCompareOp      depth_compare_op    = VK_COMPARE_OP_LESS;
	VkStencilOpState front               = {VK_STENCIL_OP_REPLACE,
                              VK_STENCIL_OP_REPLACE,
                              VK_STENCIL_OP_REPLACE,
                              VK_COMPARE_OP_ALWAYS,
                              0xff,
                              0xff,
                              1};
	VkStencilOpState back                = front;

	size_t Hash() const
	{
		size_t hash = 0;
		HashCombine(hash, depth_test_enable);
		HashCombine(hash, depth_write_enable);
		HashCombine(hash, stencil_test_enable);
		HashCombine(hash, depth_compare_op);
		HashCombine(hash, front.compareMask);
		HashCombine(hash, front.compareOp);
		HashCombine(hash, front.depthFailOp);
		HashCombine(hash, front.failOp);
		HashCombine(hash, front.passOp);
		HashCombine(hash, front.reference);
		HashCombine(hash, front.writeMask);
		HashCombine(hash, back.compareMask);
		HashCombine(hash, back.compareOp);
		HashCombine(hash, back.depthFailOp);
		HashCombine(hash, back.failOp);
		HashCombine(hash, back.passOp);
		HashCombine(hash, back.reference);
		HashCombine(hash, back.writeMask);
		return hash;
	}
};

struct ViewportState
{
	uint32_t viewport_count = 1;
	uint32_t scissor_count  = 1;

	size_t Hash() const
	{
		size_t hash = 0;
		HashCombine(hash, viewport_count);
		HashCombine(hash, scissor_count);
		return hash;
	}
};

struct MultisampleState
{
	VkSampleCountFlagBits sample_count = VK_SAMPLE_COUNT_1_BIT;

	size_t Hash() const
	{
		size_t hash = 0;
		HashCombine(hash, sample_count);
		return hash;
	}
};

struct DynamicState
{
	std::vector<VkDynamicState> dynamic_states;

	size_t Hash() const
	{
		size_t hash = 0;
		for (auto &state : dynamic_states)
		{
			HashCombine(hash, state);
		}
		return hash;
	}
};

struct VertexInputState
{
	std::vector<VkVertexInputBindingDescription>   binding_descriptions;
	std::vector<VkVertexInputAttributeDescription> attribute_descriptions;

	size_t Hash() const
	{
		size_t hash = 0;
		for (auto &description : binding_descriptions)
		{
			HashCombine(hash, description.binding);
			HashCombine(hash, description.inputRate);
			HashCombine(hash, description.stride);
		}
		for (auto &description : attribute_descriptions)
		{
			HashCombine(hash, description.binding);
			HashCombine(hash, description.format);
			HashCombine(hash, description.location);
			HashCombine(hash, description.offset);
		}
		return hash;
	}
};

struct ColorBlendAttachmentState
{
	bool                  blend_enable           = true;
	VkBlendFactor         src_color_blend_factor = VK_BLEND_FACTOR_SRC_ALPHA;
	VkBlendFactor         dst_color_blend_factor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	VkBlendOp             color_blend_op         = VK_BLEND_OP_ADD;
	VkBlendFactor         src_alpha_blend_factor = VK_BLEND_FACTOR_ONE;
	VkBlendFactor         dst_alpha_blend_factor = VK_BLEND_FACTOR_ZERO;
	VkBlendOp             alpha_blend_op         = VK_BLEND_OP_ADD;
	VkColorComponentFlags color_write_mask       = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

	size_t Hash() const
	{
		size_t hash = 0;

		HashCombine(hash, blend_enable);
		HashCombine(hash, src_color_blend_factor);
		HashCombine(hash, dst_color_blend_factor);
		HashCombine(hash, color_blend_op);
		HashCombine(hash, src_alpha_blend_factor);
		HashCombine(hash, dst_alpha_blend_factor);
		HashCombine(hash, alpha_blend_op);
		HashCombine(hash, color_write_mask);

		return hash;
	}
};

struct ColorBlendState
{
	std::vector<ColorBlendAttachmentState> attachment_states;

	size_t Hash() const
	{
		size_t hash = 0;

		for (auto& state : attachment_states)
		{
			HashCombine(hash, state.Hash());
		}

		return hash;
	}
};

class PipelineState
{
  public:
	PipelineState()  = default;
	~PipelineState() = default;

	PipelineState &SetInputAssemblyState(const InputAssemblyState &input_assembly_state);
	PipelineState &SetRasterizationState(const RasterizationState &rasterization_state);
	PipelineState &SetDepthStencilState(const DepthStencilState &depth_stencil_state);
	PipelineState &SetViewportState(const ViewportState &viewport_state);
	PipelineState &SetMultisampleState(const MultisampleState &multisample_state);
	PipelineState &SetDynamicState(const DynamicState &dynamic_state);
	PipelineState &SetVertexInputState(const VertexInputState &vertex_input_state);
	PipelineState &SetColorBlendState(const ColorBlendState &color_blend_state);
	PipelineState &LoadShader(const ShaderDesc &desc);

	const InputAssemblyState &GetInputAssemblyState() const;
	const RasterizationState &GetRasterizationState() const;
	const DepthStencilState  &GetDepthStencilState() const;
	const ViewportState      &GetViewportState() const;
	const MultisampleState   &GetMultisampleState() const;
	const DynamicState       &GetDynamicState() const;
	const VertexInputState   &GetVertexInputState() const;
	const ColorBlendState    &GetColorBlendState() const;

	const std::vector<ShaderDesc> &GetShaders() const;

	VkPipelineBindPoint GetBindPoint() const;

	size_t Hash();

  private:
	InputAssemblyState m_input_assembly_state = {};
	RasterizationState m_rasterization_state  = {};
	DepthStencilState  m_depth_stencil_state  = {};
	ViewportState      m_viewport_state       = {};
	MultisampleState   m_multisample_state    = {};
	DynamicState       m_dynamic_state        = {};
	VertexInputState   m_vertex_input_state   = {};
	ColorBlendState    m_color_blend_state    = {};

	std::vector<ShaderDesc> m_shaders;

	VkPipelineBindPoint m_bind_point = VK_PIPELINE_BIND_POINT_MAX_ENUM;

	bool m_dirty = false;

	size_t m_hash = 0;
};
}        // namespace Ilum