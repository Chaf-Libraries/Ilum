#pragma once

#include "Vulkan.hpp"

#include <map>

namespace Ilum::Vulkan
{
struct ReflectionData;
class Shader;
class RenderPass;

struct InputAssemblyState
{
	VkPrimitiveTopology topology                 = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	bool                primitive_restart_enable = false;
};

struct RasterizationState
{
	bool            depth_clamp_enable         = false;
	bool            rasterizer_discard_enable  = false;
	VkPolygonMode   polygon_mode               = VK_POLYGON_MODE_FILL;
	VkCullModeFlags cull_mode                  = VK_CULL_MODE_BACK_BIT;
	VkFrontFace     front_face                 = VK_FRONT_FACE_CLOCKWISE;
	bool            depth_bias_enable          = false;
	float           depth_bias_constant_factor = 0.f;
	float           depth_bias_clamp           = 0.f;
	float           depth_bias_slope_factor    = 0.f;
	float           line_width                 = 1.f;
};

struct ColorBlendAttachmentState
{
	bool                  blend_enable           = false;
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
	bool                                   logic_op_enable   = false;
	VkLogicOp                              logic_op          = VK_LOGIC_OP_COPY;
	float                                  blendConstants[4] = {0.f, 0.f, 0.f, 0.f};
	std::vector<ColorBlendAttachmentState> color_blend_attachments;
};

struct DepthStencilState
{
	bool             depth_test_enable        = true;
	bool             depth_write_enable       = true;
	VkCompareOp      depth_compare_op         = VK_COMPARE_OP_LESS;
	bool             depth_bounds_test_enable = false;
	bool             stencil_test_enable      = false;
	VkStencilOpState front                    = {VK_STENCIL_OP_REPLACE,
                              VK_STENCIL_OP_REPLACE,
                              VK_STENCIL_OP_REPLACE,
                              VK_COMPARE_OP_ALWAYS,
                              0xff,
                              0xff,
                              1};
	VkStencilOpState back                     = front;
	float            min_depth_bounds         = 0.f;
	float            max_depth_bounds         = 0.f;
};

struct ViewportState
{
	std::vector<VkViewport> viewports;
	std::vector<VkRect2D>   scissors;
};

struct MultisampleState
{
	VkSampleCountFlagBits     rasterization_samples = VK_SAMPLE_COUNT_1_BIT;
	bool                      sample_shading_enable = false;
	float                     min_sample_shading    = 0.f;
	std::vector<VkSampleMask> sample_mask;
	bool                      alpha_to_coverage_enable = false;
	bool                      alpha_to_one_enable      = false;
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

struct ShaderState
{
	const Shader *                          shader = nullptr;
	std::vector<VkSpecializationInfo> specializations;
};

struct ShaderStageState
{
	std::vector<ShaderState> shaders;
};

class PipelineState
{
  public:
	PipelineState()  = default;
	~PipelineState() = default;

	const InputAssemblyState &GetInputAssemblyState() const;
	const RasterizationState &GetRasterizationState() const;
	const ColorBlendState &   GetColorBlendState() const;
	const DepthStencilState & GetDepthStencilState() const;
	const ViewportState &     GetViewportState() const;
	const MultisampleState &  GetMultisampleState() const;
	const DynamicState &      GetDynamicState() const;
	const VertexInputState &  GetVertexInputState() const;
	const ShaderStageState &  GetShaderStageState() const;

	void SetInputAssemblyState(const InputAssemblyState &input_assembly_state);
	void SetRasterizationState(const RasterizationState &rasterization_state);
	void SetColorBlendState(const ColorBlendState &color_blend_state);
	void SetDepthStencilState(const DepthStencilState &depth_stencil_state);
	void SetViewportState(const ViewportState &viewport_state);
	void SetMultisampleState(const MultisampleState &multisample_state);
	void SetDynamicState(const DynamicState &dynamic_state);
	void SetVertexInputState(const VertexInputState &vertex_input_state);
	void SetShaderStageState(const ShaderStageState &shader_stage_state);

	size_t              GetHash() const;
	VkPipelineBindPoint GetBindPoint() const;

  private:
	void UpdateHash();

  private:
	InputAssemblyState m_input_assembly_state = {};
	RasterizationState m_rasterization_state  = {};
	ColorBlendState    m_color_blend_state    = {};
	DepthStencilState  m_depth_stencil_state  = {};
	ViewportState      m_viewport_state       = {};
	MultisampleState   m_multisample_state    = {};
	DynamicState       m_dynamic_state        = {};
	VertexInputState   m_vertex_input_state   = {};
	ShaderStageState   m_shader_stage_state   = {};

	size_t m_hash = 0;
};

class PipelineLayout
{
  public:
	PipelineLayout(const ReflectionData &reflection_data);
	~PipelineLayout();

	PipelineLayout(const PipelineLayout &) = delete;
	PipelineLayout &operator=(const PipelineLayout &) = delete;
	PipelineLayout(PipelineLayout &&)                 = delete;
	PipelineLayout &operator=(PipelineLayout &&) = delete;

	operator const VkPipelineLayout &() const;

	const VkPipelineLayout &GetHandle() const;

	void SetName(const std::string &name) const;

  private:
	VkPipelineLayout m_handle = VK_NULL_HANDLE;
};

class Pipeline
{
  public:
	// Create Graphics Pipeline
	Pipeline(const PipelineState &pso, const PipelineLayout &pipeline_layout, const RenderPass &render_pass, VkPipelineCache pipeline_cache = VK_NULL_HANDLE, uint32_t subpass_index = 0);
	// Create Compute Pipeline
	Pipeline(const PipelineState &pso, const PipelineLayout &pipeline_layout, VkPipelineCache pipeline_cache = VK_NULL_HANDLE);
	~Pipeline();

	Pipeline(const Pipeline &) = delete;
	Pipeline &operator=(const Pipeline &) = delete;
	Pipeline(Pipeline &&)                 = delete;
	Pipeline &operator=(Pipeline &&) = delete;

	operator const VkPipeline &() const;

	const VkPipeline &GetHandle() const;

	void SetName(const std::string &name) const;

  private:
	VkPipeline m_handle = VK_NULL_HANDLE;
};

class PipelineCache
{
  public:
	PipelineCache()  = default;
	~PipelineCache() = default;

	const PipelineLayout &RequestPipelineLayout(const PipelineState &pso);
	const Pipeline &      RequestPipeline(const PipelineState &pso, const RenderPass &render_pass, uint32_t subpass_index = 0);
	const Pipeline &      RequestPipeline(const PipelineState &pso);

  private:
	VkPipelineCache m_handle = VK_NULL_HANDLE;

	std::mutex m_pipeline_mutex;
	std::mutex m_layout_mutex;

	std::map<size_t, std::unique_ptr<PipelineLayout>> m_pipeline_layouts;
	std::map<size_t, std::unique_ptr<Pipeline>>       m_pipelines;
};
}        // namespace Ilum::Vulkan