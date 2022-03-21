#include "KullaContyAverage.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Graphics/Vulkan/VK_Debugger.h"

#include "ImGui/ImGuiContext.hpp"

#include <imgui.h>

namespace Ilum::pass
{
void KullaContyAverage::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/KullaContyEnergyAverage.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 0, "EmuLut", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	state.declareAttachment("EavgLut", VK_FORMAT_R16_SFLOAT, 1024, 1024);
	state.addOutputAttachment("EavgLut", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 1, "EavgLut", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void KullaContyAverage::resolveResources(ResolveState &resolve)
{
}

void KullaContyAverage::render(RenderPassState &state)
{
	if (!m_finish)
	{
		auto &cmd_buffer = state.command_buffer;

		vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

		for (auto &descriptor_set : state.pass.descriptor_sets)
		{
			vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
		}

		vkCmdDispatch(cmd_buffer, 1024 / 32, 1, 1);

		m_finish = true;
	}
}

void KullaContyAverage::onImGui()
{
	const auto &EavgLut = Renderer::instance()->getRenderGraph()->getAttachment("EavgLut");
	ImGui::Text("Kulla Conty Energy Average Eavg Precompute Result: ");
	ImGui::Image(ImGuiContext::textureID(EavgLut.getView(), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(100, 100));
}
}        // namespace Ilum::pass