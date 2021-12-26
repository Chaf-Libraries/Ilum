#include "Profiler.hpp"

#include "Buffer/Buffer.h"
#include "Command/CommandBuffer.hpp"
#include "GraphicsContext.hpp"

#include "Device/LogicalDevice.hpp"
#include "Device/Swapchain.hpp"

namespace Ilum
{
Profiler::Profiler()
{
	for (uint32_t i = 0; i < GraphicsContext::instance()->getSwapchain().getImageCount(); i++)
	{
		VkQueryPoolCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		createInfo.pNext = nullptr;
		createInfo.flags = 0;

		createInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
		createInfo.queryCount = 256;
		VkQueryPool handle    = VK_NULL_HANDLE;
		vkCreateQueryPool(GraphicsContext::instance()->getLogicalDevice(), &createInfo, nullptr, &handle);
		m_query_pools.push_back(handle);

		m_buffers.emplace_back(sizeof(uint32_t) * 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
	}

	m_samples.resize(GraphicsContext::instance()->getSwapchain().getImageCount());
}

Profiler ::~Profiler()
{
	for (auto &query_pool : m_query_pools)
	{
		vkDestroyQueryPool(GraphicsContext::instance()->getLogicalDevice(), query_pool, nullptr);
	}
}

void Profiler::beginFrame(const CommandBuffer &cmd_buffer)
{
	m_current_index = 0;
	vkCmdResetQueryPool(cmd_buffer, m_query_pools[GraphicsContext::instance()->getFrameIndex()], 0, 256);
}

void Profiler::beginSample(const std::string &name, const CommandBuffer &cmd_buffer)
{
	uint32_t idx = GraphicsContext::instance()->getFrameIndex();
	if (m_samples[idx].find(name) == m_samples[idx].end())
	{
		m_samples[idx][name] = Sample();
	}
	m_samples[idx][name].name = name;
	auto &start_sample        = m_samples[idx][name].start;
	start_sample.cpu_time     = std::chrono::high_resolution_clock::now();
	start_sample.index        = m_current_index++;
	vkCmdWriteTimestamp(cmd_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_query_pools[idx], start_sample.index);
	m_samples[idx][name].has_gpu = true;
}

void Profiler::beginSample(const std::string &name)
{
	uint32_t idx = GraphicsContext::instance()->getFrameIndex();
	if (m_samples[idx].find(name) == m_samples[idx].end())
	{
		m_samples[idx][name] = Sample();
	}
	m_samples[idx][name].name = name;
	auto &start_sample        = m_samples[idx][name].start;
	start_sample.cpu_time     = std::chrono::high_resolution_clock::now();
	start_sample.index        = m_current_index++;
}

void Profiler::endSample(const std::string &name, const CommandBuffer &cmd_buffer)
{
	uint32_t idx = GraphicsContext::instance()->getFrameIndex();
	if (m_samples[idx].find(name) == m_samples[idx].end())
	{
		m_samples[idx][name] = Sample();
	}

	auto &end_sample    = m_samples[idx][name].end;
	end_sample.cpu_time = std::chrono::high_resolution_clock::now();
	end_sample.index    = m_current_index++;
	vkCmdWriteTimestamp(cmd_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_query_pools[idx], end_sample.index);
}

void Profiler::endSample(const std::string &name)
{
	uint32_t idx = GraphicsContext::instance()->getFrameIndex();
	if (m_samples[idx].find(name) == m_samples[idx].end())
	{
		m_samples[idx][name] = Sample();
	}

	auto &end_sample    = m_samples[idx][name].end;
	end_sample.cpu_time = std::chrono::high_resolution_clock::now();
	end_sample.index    = m_current_index++;
}

std::map<std::string, std::pair<float, float>> Profiler::getResult() const
{
	uint32_t idx = (GraphicsContext::instance()->getFrameIndex() + 2) % 3;

	std::map<std::string, std::pair<float, float>> result;

	for (auto &[name, sample] : m_samples[idx])
	{
		uint64_t start = 0, end = 0;
		if (sample.has_gpu)
		{
			vkGetQueryPoolResults(GraphicsContext::instance()->getLogicalDevice(), m_query_pools[idx], sample.start.index, 1, sizeof(uint64_t), &start, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
			vkGetQueryPoolResults(GraphicsContext::instance()->getLogicalDevice(), m_query_pools[idx], sample.end.index, 1, sizeof(uint64_t), &end, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
		}
		result[sample.name] = std::make_pair(
		    static_cast<float>(std::chrono::duration<double, std::milli>(sample.end.cpu_time - sample.start.cpu_time).count()),
		    static_cast<float>(end - start) / 1000000.f);
	}

	return result;
}
}        // namespace Ilum