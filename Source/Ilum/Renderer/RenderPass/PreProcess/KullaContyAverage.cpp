#include "KullaContyAverage.hpp"

#include "Renderer/Renderer.hpp"

#include "Graphics/Vulkan/VK_Debugger.h"

#include "ImGui/ImGuiContext.hpp"

#include <imgui.h>

namespace Ilum::pass
{
KullaContyAverage::KullaContyAverage()
{
	m_kulla_conty_average = Image(1024, 1024, VK_FORMAT_R16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	VK_Debugger::setName(m_kulla_conty_average, "m_kulla_conty_average");
	VK_Debugger::setName(m_kulla_conty_average.getView(), "m_kulla_conty_average");
	{
		CommandBuffer cmd_buffer;
		cmd_buffer.begin();
		cmd_buffer.transferLayout(m_kulla_conty_average, VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM, VK_IMAGE_USAGE_SAMPLED_BIT);
		cmd_buffer.end();
		cmd_buffer.submitIdle();
	}
}

void KullaContyAverage::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/KullaContyEnergyAverage.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 0, "EmuLut", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 1, "EavgLut", ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void KullaContyAverage::resolveResources(ResolveState &resolve)
{
	resolve.resolve("EavgLut", m_kulla_conty_average);
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
	if (m_kulla_conty_average)
	{
		ImGui::Text("Kulla Conty Energy Average Eavg Precompute Result: ");
		ImGui::Image(ImGuiContext::textureID(m_kulla_conty_average.getView(), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(100, 100));
	}
}
}        // namespace Ilum::pass