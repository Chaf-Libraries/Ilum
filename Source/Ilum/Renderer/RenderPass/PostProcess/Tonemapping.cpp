#include "Tonemapping.hpp"

#include "Renderer/Renderer.hpp"

#include <bitset>

#include <imgui.h>

namespace Ilum::pass
{
Tonemapping::Tonemapping(const std::string &from, const std::string &to) :
    m_from(from), m_to(to)
{
}

void Tonemapping::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PostProcess/Tonemapping.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL);

	state.descriptor_bindings.bind(0, 0, m_from, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	state.declareAttachment(m_to, VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.addOutputAttachment(m_to, AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 1, m_to, ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
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

	vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_push_data), &m_push_data);

	vkCmdDispatch(cmd_buffer, (extent.width + 8 - 1) / 8, (extent.height + 8 - 1) / 8, 1);
}

void Tonemapping::onImGui()
{
	std::bitset<8> b(m_push_data.autoExposure);
	bool           autoExposure = b.test(0);

	ImGui::Checkbox("Auto Exposure", &autoExposure);
	ImGui::SliderFloat("Exposure", &m_push_data.avgLum, 0.001f, 5.0f, "%.3f");
	ImGui::SliderFloat("Brightness", &m_push_data.brightness, 0.0f, 2.0f, "%.3f");
	ImGui::SliderFloat("Contrast", &m_push_data.contrast, 0.0f, 2.0f, "%.3f");
	ImGui::SliderFloat("Saturation", &m_push_data.saturation, 0.0f, 5.0f, "%.3f");
	ImGui::SliderFloat("Vignette", &m_push_data.vignette, 0.0f, 2.0f, "%.3f");

	if (autoExposure)
	{
		bool localExposure = b.test(1);
		if (ImGui::TreeNode("Auto Settings"))
		{
			ImGui::Checkbox("Local", &localExposure);
			ImGui::SliderFloat("Burning White", &m_push_data.Ywhite, 0.f, 1.f, "%.3f");
			ImGui::SliderFloat("Brightness", &m_push_data.key, 0.f, 1.f, "%.3f");
			b.set(1, localExposure);
			ImGui::End();
		}
	}

	b.set(0, autoExposure);
	m_push_data.autoExposure = b.to_ulong();
}
}        // namespace Ilum::pass