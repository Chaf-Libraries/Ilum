#pragma once

#include "Graphics/Vulkan.hpp"

namespace Ilum::Graphics
{
class PipelineState;
class PipelineLayout;
class RenderPass;
class Device;

class Pipeline
{
  public:
	// Create Graphics Pipeline
	Pipeline(const Device& device, const PipelineState &pso, const PipelineLayout &pipeline_layout, const RenderPass &render_pass, VkPipelineCache pipeline_cache = VK_NULL_HANDLE, uint32_t subpass_index = 0);
	// Create Compute Pipeline
	Pipeline(const Device &device, const PipelineState &pso, const PipelineLayout &pipeline_layout, VkPipelineCache pipeline_cache = VK_NULL_HANDLE);
	// TODO: Create RTX Pipeline
	~Pipeline();

	Pipeline(const Pipeline &) = delete;
	Pipeline &operator=(const Pipeline &) = delete;
	Pipeline(Pipeline &&)                 = delete;
	Pipeline &operator=(Pipeline &&) = delete;

	operator const VkPipeline &() const;

	const VkPipeline &GetHandle() const;

  private:
	const Device & m_device;
	VkPipeline m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum::Graphics