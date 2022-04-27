#include "FXAA.hpp"

#include "Renderer/Renderer.hpp"

#include <imgui.h>

namespace Ilum::pass
{
FXAA::FXAA(const std::string &input, const std::string &output, const std::string &quality) :
    m_input(input), m_output(output), m_quality(quality)
{
}

void FXAA::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PostProcess/FXAA.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL, "main", {m_quality});

	state.declareAttachment(m_output, VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.addOutputAttachment(m_output, AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 0, m_input, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 1, m_output, ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void FXAA::resolveResources(ResolveState &resolve)
{
}

void FXAA::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	auto &extent = Renderer::instance()->getRenderTargetExtent();

	vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_push_constants), &m_push_constants);
	vkCmdDispatch(cmd_buffer, (extent.width + 8 - 1) / 8, (extent.height + 8 - 1) / 8, 1);
}

void FXAA::onImGui()
{
	ImGui::Text("%s", m_quality.c_str());
	ImGui::SliderFloat("Fixed Threshold", &m_push_constants.fixed_threshold, 0.0312f, 0.0833f, "%.4f");
	ImGui::SliderFloat("Relative Threshold", &m_push_constants.relative_threshold, 0.063f, 0.333f, "%.4f");
	ImGui::SliderFloat("Subpixel Blending", &m_push_constants.subpixel_blending, 0.f, 1.f, "%.2f");
}
}        // namespace Ilum::pass