#include "RenderPass.hpp"
#include "RenderGraph.hpp"

#include "Device/LogicalDevice.hpp"
#include "Device/Swapchain.hpp"

#include "Graphics/GraphicsContext.hpp"

#include <imgui.h>

namespace Ilum
{
const Image &RenderPassState::getAttachment(const std::string &name)
{
	return graph.getAttachment(name);
}

void RenderPass::beginProfile(RenderPassState &state)
{
	m_thread_id = std::this_thread::get_id();

	uint32_t idx = GraphicsContext::instance()->getFrameIndex();

	auto result  = vkGetQueryPoolResults(GraphicsContext::instance()->getLogicalDevice(), state.pass.query_pools[(idx + 1) % GraphicsContext::instance()->getSwapchain().getImageCount()], 0, 1, sizeof(uint64_t), &m_gpu_start, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
	auto result1 = vkGetQueryPoolResults(GraphicsContext::instance()->getLogicalDevice(), state.pass.query_pools[(idx + 1) % GraphicsContext::instance()->getSwapchain().getImageCount()], 1, 1, sizeof(uint64_t), &m_gpu_end, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
	m_gpu_time = static_cast<float>(m_gpu_end - m_gpu_start) / 1000000.f;
	m_cpu_time = std::chrono::duration<float, std::milli>(m_cpu_end - m_cpu_start).count();

	vkCmdResetQueryPool(state.command_buffer, state.pass.query_pools[idx], 0, 2);
	vkCmdWriteTimestamp(state.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, state.pass.query_pools[idx], 0);
	m_cpu_start = std::chrono::high_resolution_clock::now();
}

void RenderPass::endProfile(RenderPassState &state)
{
	uint32_t idx = GraphicsContext::instance()->getFrameIndex();
	vkCmdWriteTimestamp(state.command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, state.pass.query_pools[idx], 1);
	m_cpu_end = std::chrono::high_resolution_clock::now();
}

float RenderPass::getGPUTime()
{
	return m_gpu_time;
}

float RenderPass::getCPUTime()
{
	return m_cpu_time;
}

size_t RenderPass::getThreadID()
{
	return std::hash<std::thread::id>{}(m_thread_id);
}
}        // namespace Ilum