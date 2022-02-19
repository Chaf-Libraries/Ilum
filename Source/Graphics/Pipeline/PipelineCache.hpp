#pragma once

#include "Graphics/Vulkan.hpp"

#include <map>

namespace Ilum::Graphics
{
class Device;
class PipelineLayout;
class Pipeline;
class RenderPass;
class PipelineState;
struct ReflectionData;

class PipelineCache
{
  public:
	PipelineCache(const Device &device);
	~PipelineCache();

	operator const VkPipelineCache &() const;

	const VkPipelineCache &GetHandle() const;

	Pipeline &RequestPipeline(const PipelineState &pso, const PipelineLayout& layout, const RenderPass &render_pass, uint32_t subpass_index = 0);
	Pipeline &RequestPipeline(const PipelineState &pso, const PipelineLayout &layout);

	PipelineLayout &RequestPipelineLayout(const ReflectionData &reflection_data, const std::vector<VkDescriptorSetLayout> &descriptor_set_layouts);

  private:
	const Device &m_device;

	std::mutex m_pipeline_mutex;
	std::mutex m_layout_mutex;

	std::map<size_t, std::unique_ptr<Pipeline>>       m_pipelines;
	std::map<size_t, std::unique_ptr<PipelineLayout>> m_pipeline_layouts;

	VkPipelineCache m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum::Graphics