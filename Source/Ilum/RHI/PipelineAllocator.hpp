#pragma once

#include "FrameBuffer.hpp"
#include "PipelineState.hpp"
#include "ShaderBindingTable.hpp"

namespace Ilum
{
class RHIDevice;

struct SBT
{
	std::unique_ptr<ShaderBindingTable> raygen   = nullptr;
	std::unique_ptr<ShaderBindingTable> miss     = nullptr;
	std::unique_ptr<ShaderBindingTable> hit      = nullptr;
	std::unique_ptr<ShaderBindingTable> callable = nullptr;
};

class PipelineAllocator
{
  public:
	PipelineAllocator(RHIDevice *device);
	~PipelineAllocator();

	VkPipelineLayout CreatePipelineLayout(PipelineState &pso);
	VkRenderPass     CreateRenderPass(FrameBuffer& frame_buffer);
	VkFramebuffer    CreateFrameBuffer(FrameBuffer &frame_buffer);
	VkPipeline       CreatePipeline(PipelineState &pso);
	SBT             &CreateShaderBindingTable(PipelineState &pso, VkPipeline pipeline);

  private:
	VkPipeline CreateComputePipeline(PipelineState &pso);
	VkPipeline CreateGraphicsPipeline(PipelineState &pso);
	VkPipeline CreateRayTracingPipeline(PipelineState &pso);

  private:
	RHIDevice *p_device = nullptr;

	std::map<size_t, VkPipeline>       m_pipelines;
	std::map<size_t, VkRenderPass>             m_render_pass;
	std::map<size_t, VkFramebuffer>            m_frame_buffer;
	std::map<size_t, VkPipelineLayout> m_pipeline_layout;
	std::map<VkPipeline, std::unique_ptr<SBT>> m_sbt;
};
}        // namespace Ilum