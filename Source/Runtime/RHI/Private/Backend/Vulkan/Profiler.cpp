#include "Profiler.hpp"

#include "Command.hpp"
#include "Device.hpp"

namespace Ilum::Vulkan
{
Profiler::Profiler(RHIDevice *device, uint32_t frame_count) :
    RHIProfiler(device, frame_count)
{
	VkQueryPoolCreateInfo createInfo = {};
	createInfo.sType                 = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	createInfo.pNext                 = nullptr;
	createInfo.flags                 = 0;

	createInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
	createInfo.queryCount = 2;

	m_query_pools.resize(frame_count);
	for (auto &pool : m_query_pools)
	{
		vkCreateQueryPool(static_cast<Device *>(p_device)->GetDevice(), &createInfo, nullptr, &pool);
	}
}

Profiler ::~Profiler()
{
	for (auto &pool : m_query_pools)
	{
		vkDestroyQueryPool(static_cast<Device *>(p_device)->GetDevice(), pool, nullptr);
	}
}

void Profiler::Begin(RHICommand *cmd_buffer, uint32_t frame_index)
{
	m_current_index = frame_index;
	m_cmd_buffer    = static_cast<Command *>(cmd_buffer)->GetHandle();

	m_state.thread_id = std::this_thread::get_id();

	vkGetQueryPoolResults(static_cast<Device *>(p_device)->GetDevice(), m_query_pools[m_current_index], 0, 1, sizeof(uint64_t), &m_state.gpu_start, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
	vkGetQueryPoolResults(static_cast<Device *>(p_device)->GetDevice(), m_query_pools[m_current_index], 1, 1, sizeof(uint64_t), &m_state.gpu_end, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
	m_state.gpu_time = static_cast<float>(m_state.gpu_end - m_state.gpu_start) / 1000000.f;
	m_state.cpu_time = std::chrono::duration<float, std::milli>(m_state.cpu_end - m_state.cpu_start).count();

	vkCmdResetQueryPool(m_cmd_buffer, m_query_pools[m_current_index], 0, 2);
	vkCmdWriteTimestamp(m_cmd_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_query_pools[m_current_index], 0);
	m_state.cpu_start = std::chrono::high_resolution_clock::now();
}

void Profiler::End(RHICommand *cmd_buffer)
{
	ASSERT(m_cmd_buffer != VK_NULL_HANDLE);
	vkCmdWriteTimestamp(m_cmd_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_query_pools[m_current_index], 1);
	m_state.cpu_end = std::chrono::high_resolution_clock::now();
}
}        // namespace Ilum::Vulkan