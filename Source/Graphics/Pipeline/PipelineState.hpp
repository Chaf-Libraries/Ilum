#pragma once

#include "Utils/PCH.hpp"

#include "Graphics/Descriptor/DescriptorBinding.hpp"
#include "Graphics/Pipeline/Shader.hpp"

#include "Math/Vector4.h"

namespace Ilum
{
class Shader;

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

//struct PipelineState
//{
//	InputAssemblyState input_assembly_state;
//	RasterizationState rasterization_state;
//	DepthStencilState  depth_stencil_state;
//	ViewportState      viewport_state;
//	MultisampleState   multisample_state;
//	DynamicState       dynamic_state;
//	VertexInputState   vertex_input_state;
//
//	PipelineState();
//
//	~PipelineState() = default;
//};

enum class AttachmentState
{
	Discard_Color,
	Discard_Depth_Stencil,
	Load_Color,
	Load_Depth_Stencil,
	Clear_Color,
	Clear_Depth_Stencil
};

class PipelineState
{
  public:
	struct ImageDependency
	{
		std::string          name;
		VkImageUsageFlagBits usage;
	};

	struct BufferDependency
	{
		std::string           name;
		VkBufferUsageFlagBits usage;
	};

	struct AttachmentDeclaration
	{
		std::string name;
		VkFormat    format;
		uint32_t    width;
		uint32_t    height;
		bool        mipmaps = 0;
		uint32_t    layers  = 1;
	};

	struct OutputAttachment
	{
		constexpr static uint32_t ALL_LAYERS = std::numeric_limits<uint32_t>::max();

		std::string              name;
		VkClearColorValue        color_clear;
		VkClearDepthStencilValue depth_stencil_clear;
		AttachmentState          state;
		uint32_t                 layer;
	};

  public:
	PipelineState() = default;

	~PipelineState() = default;

	PipelineState &addOutputAttachment(const std::string &name, VkClearColorValue clear, uint32_t layer = OutputAttachment::ALL_LAYERS);
	PipelineState &addOutputAttachment(const std::string &name, VkClearDepthStencilValue clear, uint32_t layer = OutputAttachment::ALL_LAYERS);
	PipelineState &addOutputAttachment(const std::string &name, AttachmentState state, uint32_t layer = OutputAttachment::ALL_LAYERS);

	PipelineState &addDependency(const std::string &name, VkBufferUsageFlagBits usage);
	PipelineState &addDependency(const std::string &name, VkImageUsageFlagBits usage);

	PipelineState &declareAttachment(const std::string &name, VkFormat format, uint32_t width = 0, uint32_t height = 0, bool mipmaps = false, uint32_t layers = 1);

	const std::vector<BufferDependency> &     getBufferDependencies() const;
	const std::vector<ImageDependency> &      getImageDependencies() const;
	const std::vector<AttachmentDeclaration> &getAttachmentDeclarations() const;
	const std::vector<OutputAttachment> &     getOutputAttachments() const;

  public:
	InputAssemblyState input_assembly_state = {};

	RasterizationState rasterization_state = {};

	DepthStencilState depth_stencil_state = {};

	ViewportState viewport_state = {};

	MultisampleState multisample_state = {};

	DynamicState dynamic_state = {};

	VertexInputState vertex_input_state = {};

	ColorBlendAttachmentState color_blend_attachment_state = {};

	DescriptorBinding descriptor_bindings;

	Shader shader;

  private:
	std::vector<BufferDependency> m_buffer_dependencies;
	std::vector<ImageDependency>  m_image_dependencies;

	std::vector<AttachmentDeclaration> m_attachment_declarations;
	std::vector<OutputAttachment>      m_output_attachments;
};

}        // namespace Ilum