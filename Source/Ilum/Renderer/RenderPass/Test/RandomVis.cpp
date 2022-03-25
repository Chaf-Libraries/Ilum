#include "RandomVis.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Vulkan/VK_Debugger.h"

#include "Device/LogicalDevice.hpp"

#include "ImGui/ImGuiContext.hpp"

#include <imgui.h>

namespace Ilum::pass
{
void RandomVis::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Test/RandomVis.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.declareAttachment("RandomVis", VK_FORMAT_R8G8B8A8_UNORM, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.addOutputAttachment("RandomVis", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 0, "RandomVis", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void RandomVis::resolveResources(ResolveState &resolve)
{
}

void RandomVis::render(RenderPassState &state)
{
	if (m_update)
	{
		auto &cmd_buffer = state.command_buffer;

		vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

		for (auto &descriptor_set : state.pass.descriptor_sets)
		{
			vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
		}

		m_push_data.extent = Renderer::instance()->getRenderTargetExtent();
		m_push_data.frame++;

		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_push_data), &m_push_data);

		vkCmdDispatch(cmd_buffer, 1, 1, 1);

		m_update = false;
	}
}

void RandomVis::onImGui()
{
	if (ImGui::Button("Clear"))
	{
		const auto &RandomVis = Renderer::instance()->getRenderGraph()->getAttachment("RandomVis");
		GraphicsContext::instance()->getQueueSystem().waitAll();
		CommandBuffer cmd_buffer;
		cmd_buffer.begin();
		VkClearColorValue clear_color = {};
		clear_color.uint32[0]         = 0;
		clear_color.uint32[1]         = 0;
		clear_color.uint32[2]         = 0;
		clear_color.uint32[3]         = std::numeric_limits<uint32_t>::max();
		clear_color.int32[0]           = 0;
		clear_color.int32[1]          =0;
		clear_color.int32[2]          =0;
		clear_color.int32[3]          = std::numeric_limits<int32_t>::max();
		clear_color.float32[0]         = 0;
		clear_color.float32[1]         = 0;
		clear_color.float32[2]         = 0;
		clear_color.float32[3]         = std::numeric_limits<float>::max();
		cmd_buffer.transferLayout(RandomVis, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		vkCmdClearColorImage(cmd_buffer, RandomVis, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &RandomVis.getSubresourceRange());
		cmd_buffer.transferLayout(RandomVis, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_USAGE_SAMPLED_BIT);
		cmd_buffer.end();
		cmd_buffer.submitIdle();
	}

	if (ImGui::Button("Update"))
	{
		m_update = true;

		const auto &RandomVis = Renderer::instance()->getRenderGraph()->getAttachment("RandomVis");
		GraphicsContext::instance()->getQueueSystem().waitAll();
		CommandBuffer cmd_buffer;
		cmd_buffer.begin();
		VkClearColorValue clear_color = {};
		clear_color.uint32[0]         = 0;
		clear_color.uint32[1]         = 0;
		clear_color.uint32[2]         = 0;
		clear_color.uint32[3]         = std::numeric_limits<uint32_t>::max();
		clear_color.int32[0]          = 0;
		clear_color.int32[1]          = 0;
		clear_color.int32[2]          = 0;
		clear_color.int32[3]          = std::numeric_limits<int32_t>::max();
		clear_color.float32[0]        = 0;
		clear_color.float32[1]        = 0;
		clear_color.float32[2]        = 0;
		clear_color.float32[3]        = std::numeric_limits<float>::max();
		cmd_buffer.transferLayout(RandomVis, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		vkCmdClearColorImage(cmd_buffer, RandomVis, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &RandomVis.getSubresourceRange());
		cmd_buffer.transferLayout(RandomVis, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_USAGE_SAMPLED_BIT);
		cmd_buffer.end();
		cmd_buffer.submitIdle();
	}

	if (ImGui::DragInt("Sample Count", &m_push_data.m_sample_count, 1.f, 0, std::numeric_limits<int>::max()))
	{
		m_update = true;

		const auto &RandomVis = Renderer::instance()->getRenderGraph()->getAttachment("RandomVis");
		GraphicsContext::instance()->getQueueSystem().waitAll();
		CommandBuffer cmd_buffer;
		cmd_buffer.begin();
		VkClearColorValue clear_color = {};
		clear_color.uint32[0]         = 0;
		clear_color.uint32[1]         = 0;
		clear_color.uint32[2]         = 0;
		clear_color.uint32[3]         = std::numeric_limits<uint32_t>::max();
		clear_color.int32[0]          = 0;
		clear_color.int32[1]          = 0;
		clear_color.int32[2]          = 0;
		clear_color.int32[3]          = std::numeric_limits<int32_t>::max();
		clear_color.float32[0]        = 0;
		clear_color.float32[1]        = 0;
		clear_color.float32[2]        = 0;
		clear_color.float32[3]        = std::numeric_limits<float>::max();
		cmd_buffer.transferLayout(RandomVis, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		vkCmdClearColorImage(cmd_buffer, RandomVis, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &RandomVis.getSubresourceRange());
		cmd_buffer.transferLayout(RandomVis, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_USAGE_SAMPLED_BIT);
		cmd_buffer.end();
		cmd_buffer.submitIdle();
	}

	const char *const sampling_method[] = {"Uniform", "Stratified", "Halton", "Sobel"};
	ImGui::Combo("Sampling Method", &m_push_data.sampling_method, sampling_method, 4);
}
}        // namespace Ilum::pass