#pragma once

#include "../Vulkan.hpp"

#include <Shader/SpirvReflection.hpp>

#include <map>

namespace Ilum::Graphics
{
class Shader;

struct VertexInputState
{
	std::vector<VkVertexInputBindingDescription>   bindings;
	std::vector<VkVertexInputAttributeDescription> attributes;
};

struct InputAssemblyState
{
	VkPrimitiveTopology topology                 = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	bool                primitive_restart_enable = VK_FALSE;
};

struct RasterizationState
{
	bool            depth_clamp_enable        = false;
	bool            rasterizer_discard_enable = false;
	VkPolygonMode   polygon_mode              = VK_POLYGON_MODE_FILL;
	VkCullModeFlags cull_mode                 = VK_CULL_MODE_BACK_BIT;
	VkFrontFace     front_face                = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	bool            depth_bias_enable         = false;
};

struct ViewportState
{
	uint32_t viewport_count = 1;
	uint32_t scissor_count  = 1;
};

struct MultisampleState
{
	VkSampleCountFlagBits rasterization_samples    = VK_SAMPLE_COUNT_1_BIT;
	bool                  sample_shading_enable    = false;
	float                 min_sample_shading       = 0.f;
	VkSampleMask          sample_mask              = 0;
	bool                  alpha_to_coverage_enable = false;
	bool                  alpha_to_one_enable      = false;
};

struct StencilOpState
{
	VkStencilOp fail_op       = VK_STENCIL_OP_REPLACE;
	VkStencilOp pass_op       = VK_STENCIL_OP_REPLACE;
	VkStencilOp depth_fail_op = VK_STENCIL_OP_REPLACE;
	VkCompareOp compare_op    = VK_COMPARE_OP_NEVER;
};

struct DepthStencilState
{
	bool           depth_test_enable        = true;
	bool           depth_write_enable       = true;
	VkCompareOp    depth_compare_op         = VK_COMPARE_OP_GREATER;
	bool           depth_bounds_test_enable = false;
	bool           stencil_test_enable      = false;
	StencilOpState front                    = {};
	StencilOpState back                     = {};
};

struct ColorBlendAttachmentState
{
	bool                  blend_enable           = false;
	VkBlendFactor         src_color_blend_factor = VK_BLEND_FACTOR_ONE;
	VkBlendFactor         dst_color_blend_factor = VK_BLEND_FACTOR_ZERO;
	VkBlendOp             color_blend_op         = VK_BLEND_OP_ADD;
	VkBlendFactor         src_alpha_blend_factor = VK_BLEND_FACTOR_ONE;
	VkBlendFactor         dst_alpha_blend_factor = VK_BLEND_FACTOR_ZERO;
	VkBlendOp             alpha_blend_op         = VK_BLEND_OP_ADD;
	VkColorComponentFlags color_write_mask       = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
};

struct ColorBlendState
{
	bool                                   logic_op_enable = false;
	VkLogicOp                              logic_op        = VK_LOGIC_OP_CLEAR;
	std::vector<ColorBlendAttachmentState> attachments;
};

struct ShaderStageState
{
	const Shader *shader = nullptr;

	std::map<uint32_t, std::vector<uint8_t>> specialization_constants;
};

struct StageState
{
	std::vector<ShaderStageState> shader_stage_states;
};

class PipelineState
{
  public:
	PipelineState()  = default;
	~PipelineState() = default;

	void SetVertexInputState(const VertexInputState &vertex_input_state);
	void SetInputAssemblyState(const InputAssemblyState &input_assembly_state);
	void SetRasterizationState(const RasterizationState &rasterization_state);
	void SetViewportState(const ViewportState &viewport_state);
	void SetMultisampleState(const MultisampleState &multisample_state);
	void SetDepthStencilState(const DepthStencilState &depth_stencil_state);
	void SetColorBlendState(const ColorBlendState &color_blend_state);
	void SetStageState(const StageState &stage_state);
	void SetSubpassIndex(uint32_t subpass_index);

	const VertexInputState &  GetVertexInputState() const;
	const InputAssemblyState &GetInputAssemblyState() const;
	const RasterizationState &GetRasterizationState() const;
	const ViewportState &     GetViewportState() const;
	const MultisampleState &  GetMultisampleState() const;
	const DepthStencilState & GetDepthStencilState() const;
	const ColorBlendState &   GetColorBlendState() const;
	const StageState &        GetStageState() const;
	const ReflectionData &    GetReflectionData() const;
	uint32_t                  GetSubpassIndex() const;
	size_t                    GetHash() const;
	VkPipelineBindPoint       GetBindPoint() const;

  private:
	void UpdateHash();

  private:
	VertexInputState   m_vertex_input_state;
	InputAssemblyState m_input_assembly_state;
	RasterizationState m_rasterization_state;
	ViewportState      m_viewport_state;
	MultisampleState   m_multisample_state;
	DepthStencilState  m_depth_stencil_state;
	ColorBlendState    m_color_blend_state;
	StageState         m_stage_state;
	ReflectionData     m_reflection_data;
	uint32_t           m_subpass_index = 0;
	size_t             m_hash          = 0;
};
}        // namespace Ilum::Graphics