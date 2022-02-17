#include "PipelineCache.hpp"
#include "Device/Device.hpp"
#include "Pipeline.hpp"
#include "PipelineLayout.hpp"
#include "PipelineState.hpp"
#include "RenderPass/RenderPass.hpp"
#include "Shader/Shader.hpp"
#include "Shader/SpirvReflection.hpp"

#include <Core/Hash.hpp>

namespace Ilum::Graphics
{
PipelineCache::PipelineCache(const Device &device) :
    m_device(device)
{
	VkPipelineCacheCreateInfo pipeline_cache_create_info = {};
	pipeline_cache_create_info.sType                     = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	vkCreatePipelineCache(m_device, &pipeline_cache_create_info, nullptr, &m_handle);
}

PipelineCache::~PipelineCache()
{
	if (m_handle)
	{
		vkDestroyPipelineCache(m_device, m_handle, nullptr);
	}
}

PipelineCache::operator const VkPipelineCache &() const
{
	return m_handle;
}

const VkPipelineCache &PipelineCache::GetHandle() const
{
	return m_handle;
}

Pipeline &PipelineCache::RequestPipeline(const PipelineState &pso, const PipelineLayout &layout, const RenderPass &render_pass, uint32_t subpass_index)
{
	size_t hash = 0;
	Core::HashCombine(hash, pso.GetHash());
	Core::HashCombine(hash, render_pass.GetHash());
	Core::HashCombine(hash, subpass_index);

	if (m_pipelines.find(hash) == m_pipelines.end())
	{
		std::lock_guard<std::mutex> lock(m_pipeline_mutex);
		m_pipelines.emplace(hash, std::make_unique<Pipeline>(m_device, pso, layout, render_pass, m_handle, subpass_index));
	}

	return *m_pipelines.at(hash);
}

Pipeline &PipelineCache::RequestPipeline(const PipelineState &pso, const PipelineLayout &layout)
{
	size_t hash = 0;
	Core::HashCombine(hash, pso.GetHash());

	if (m_pipelines.find(hash) == m_pipelines.end())
	{
		std::lock_guard<std::mutex> lock(m_pipeline_mutex);
		m_pipelines.emplace(hash, std::make_unique<Pipeline>(m_device, pso, layout, m_handle));
	}

	return *m_pipelines.at(hash);
}

PipelineLayout &PipelineCache::RequestPipelineLayout(const ReflectionData &reflection_data, const std::vector<VkDescriptorSetLayout> &descriptor_set_layouts)
{
	if (m_pipeline_layouts.find(reflection_data.hash) == m_pipeline_layouts.end())
	{
		std::lock_guard<std::mutex> lock(m_layout_mutex);
		m_pipeline_layouts.emplace(reflection_data.hash, std::make_unique<PipelineLayout>(m_device, reflection_data, descriptor_set_layouts));
	}

	return *m_pipeline_layouts.at(reflection_data.hash);
}
}        // namespace Ilum::Graphics