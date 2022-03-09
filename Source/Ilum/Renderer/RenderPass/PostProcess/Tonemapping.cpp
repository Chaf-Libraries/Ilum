#include "Tonemapping.hpp"

#include "Renderer/Renderer.hpp"

#include <imgui.h>

namespace Ilum::pass
{
Tonemapping::Tonemapping(const std::string &result) :
    m_result(result)
{
}

void Tonemapping::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PostProcess/Tonemapping.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 0, m_result, ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void Tonemapping::resolveResources(ResolveState &resolve)
{
}

void Tonemapping::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	auto &extent = Renderer::instance()->getRenderTargetExtent();

	m_tonemapping_data.extent = extent;
	vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_tonemapping_data), &m_tonemapping_data);

	vkCmdDispatch(cmd_buffer, (extent.width + 32 - 1) / 32, (extent.height + 32 - 1) / 32, 1);
}

void Tonemapping::onImGui()
{
	ImGui::DragFloat("Exposure", &m_tonemapping_data.exposure, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
	ImGui::DragFloat("Gamma", &m_tonemapping_data.gamma, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
}
}        // namespace Ilum::pass