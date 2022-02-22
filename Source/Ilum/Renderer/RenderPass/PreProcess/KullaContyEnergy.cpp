#include "KullaContyEnergy.hpp"

#include "Renderer/Renderer.hpp"

#include "Graphics/Vulkan/VK_Debugger.h"

namespace Ilum::pass
{
KullaContyEnergy::KullaContyEnergy()
{
	m_kulla_conty_energy = Image(128, 128, VK_FORMAT_R16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	VK_Debugger::setName(m_kulla_conty_energy, "m_kulla_conty_energy");
	VK_Debugger::setName(m_kulla_conty_energy.getView(), "m_kulla_conty_energy");

	{
		CommandBuffer cmd_buffer;
		cmd_buffer.begin();
		cmd_buffer.transferLayout(m_kulla_conty_energy, VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM, VK_IMAGE_USAGE_SAMPLED_BIT);
		cmd_buffer.end();
		cmd_buffer.submitIdle();
	}
}

void KullaContyEnergy::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/PreProcess/KullaContyEnergy.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 0, "LUT - Emu", ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void KullaContyEnergy::resolveResources(ResolveState &resolve)
{
	resolve.resolve("LUT - Emu", m_kulla_conty_energy);
}

void KullaContyEnergy::render(RenderPassState &state)
{
	if (!m_finish)
	{
		auto &cmd_buffer = state.command_buffer;

		vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

		for (auto &descriptor_set : state.pass.descriptor_sets)
		{
			vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
		}

		vkCmdDispatch(cmd_buffer, 128 / 32, 128 / 32, 1);

		m_finish = true;
	}
}
}        // namespace Ilum::pass