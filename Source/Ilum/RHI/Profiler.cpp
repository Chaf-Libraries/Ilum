#include "Profiler.hpp"
#include "Command.hpp"
#include "Device.hpp"

namespace Ilum
{
Profiler::Profiler(RHIDevice *device) :
    p_device(device)
{
	VkQueryPoolCreateInfo createInfo = {};
	createInfo.sType                 = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	createInfo.pNext                 = nullptr;
	createInfo.flags                 = 0;

	createInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
	createInfo.queryCount = 2;

	m_query_pools.resize(p_device->GetSwapchainImages().size());
	for (auto &pool : m_query_pools)
	{
		vkCreateQueryPool(p_device->GetDevice(), &createInfo, nullptr, &pool);
	}
}

Profiler::~Profiler()
{
	for (auto &pool : m_query_pools)
	{
		vkDestroyQueryPool(p_device->GetDevice(), pool, nullptr);
	}
}

const ProfileState &Profiler::GetProfileState() const
{
	return m_state;
}

void Profiler::Begin(CommandBuffer &cmd_buffer)
{
	uint32_t idx = p_device->GetCurrentFrame();

	m_state.thread_id = std::this_thread::get_id();

	vkGetQueryPoolResults(p_device->GetDevice(), m_query_pools[idx], 0, 1, sizeof(uint64_t), &m_state.gpu_start, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
	vkGetQueryPoolResults(p_device->GetDevice(), m_query_pools[idx], 1, 1, sizeof(uint64_t), &m_state.gpu_end, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
	m_state.gpu_time = static_cast<float>(m_state.gpu_end - m_state.gpu_start) / 1000000.f;
	m_state.cpu_time = std::chrono::duration<float, std::milli>(m_state.cpu_end - m_state.cpu_start).count();

	vkCmdResetQueryPool(cmd_buffer, m_query_pools[idx], 0, 2);
	vkCmdWriteTimestamp(cmd_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_query_pools[idx], 0);
	m_state.cpu_start = std::chrono::high_resolution_clock::now();
}

void Profiler::End(CommandBuffer &cmd_buffer)
{
	uint32_t idx = p_device->GetCurrentFrame();
	vkCmdWriteTimestamp(cmd_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_query_pools[idx], 1);
	m_state.cpu_end = std::chrono::high_resolution_clock::now();
}

}        // namespace Ilum