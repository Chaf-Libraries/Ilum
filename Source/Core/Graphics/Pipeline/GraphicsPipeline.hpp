#pragma once

#include "Pipeline.hpp"
#include "Shader.hpp"

namespace Ilum
{
class LogicalDevice;

class GraphicsPipeline : public Pipeline
{
  public:
	struct InputAssemblyState
	{
		VkPrimitiveTopology topology                 = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		bool                primitive_restart_enable = false;
	};

	struct RasterizationState
	{
		VkPolygonMode   polygon_mode = VK_POLYGON_MODE_FILL;
		VkCullModeFlags cull_mode    = VK_CULL_MODE_BACK_BIT;
		VkFrontFace     front_face   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	};

	struct ColorBlendAttachmentState
	{
		VkColorComponentFlags color_write_mask = 0xf;
		bool                  blend_enable     = false;
	};

	struct ColorBlendState
	{
		std::vector<ColorBlendAttachmentState> color_blend_attachment_states;
	};

	struct DepthStencilState
	{
		bool        depth_test_enable  = true;
		bool        depth_write_enable = true;
		VkCompareOp depth_compare_op   = VK_COMPARE_OP_LESS_OR_EQUAL;
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
		std::vector<VkDynamicState> dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_LINE_WIDTH};
	};

	struct PipelineState
	{
		InputAssemblyState input_assembly_state = {};
		RasterizationState rasterization_state = {};
		ColorBlendState    color_blend_state   = {};
		DepthStencilState  depth_stencil_state = {};
		ViewportState      viewport_state      = {};
		MultisampleState   multisample_state   = {};
		DynamicState       dynamic_state       = {};
	};

  public:
	GraphicsPipeline(
	    const LogicalDevice &           logical_device,
	    VkPipelineCache                 pipeline_cache,
	    const std::vector<std::string> &shader_paths,
	    PipelineState                   pipeline_state = {},
	    const Shader::Variant &         variant        = {});

	~GraphicsPipeline();

  private:
	PipelineState m_pipeline_state = {};
};
}        // namespace Ilum