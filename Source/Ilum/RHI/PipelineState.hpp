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
};

struct RasterizationState
{
	VkPolygonMode   polygon_mode = VK_POLYGON_MODE_FILL;
	VkCullModeFlags cull_mode    = VK_CULL_MODE_BACK_BIT;
	VkFrontFace     front_face   = VK_FRONT_FACE_CLOCKWISE;
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
};

struct ViewportState
{
	uint32_t viewport_count = 1;
	uint32_t scissor_count  = 1;
};

struct MultisampleState
{
	VkSampleCountFlagBits sample_count = VK_SAMPLE_COUNT_1_BIT;
};

struct DynamicState
{
	std::vector<VkDynamicState> dynamic_states;
};

struct VertexInputState
{
	std::vector<VkVertexInputBindingDescription>   binding_descriptions;
	std::vector<VkVertexInputAttributeDescription> attribute_descriptions;
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
};

struct ColorBlendState
{
	std::vector<ColorBlendAttachmentState> attachment_states;
};

class PipelineState
{
	friend class CommandBuffer;

  public:
	PipelineState(RHIDevice *device);
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

	size_t Hash();

  private:
	RHIDevice *p_device;

	InputAssemblyState m_input_assembly_state = {};
	RasterizationState m_rasterization_state  = {};
	DepthStencilState  m_depth_stencil_state  = {};
	ViewportState      m_viewport_state       = {};
	MultisampleState   m_multisample_state    = {};
	DynamicState       m_dynamic_state        = {};
	VertexInputState   m_vertex_input_state   = {};
	ColorBlendState    m_color_blend_state    = {};

	ShaderReflectionData                                                                 m_shader_meta;
	std::map<VkShaderStageFlagBits, std::vector<std::pair<std::string, VkShaderModule>>> m_shaders;

	bool m_dirty = false;

	size_t m_hash = 0;
};
}        // namespace Ilum