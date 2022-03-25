#include "CubemapSHAdd.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "ImGui/ImGuiContext.hpp"

#include <imgui.h>

namespace Ilum::pass
{
void CubemapSHAdd::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/CubemapSHAdd.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 1, "SHIntermediate", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	state.declareAttachment("IrradianceSH", VK_FORMAT_R16G16B16A16_SFLOAT, 9, 1, false, 1);
	state.addOutputAttachment("IrradianceSH", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 0, "IrradianceSH", ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void CubemapSHAdd::resolveResources(ResolveState &resolve)
{
}

void CubemapSHAdd::render(RenderPassState &state)
{
	if (!m_update && !Renderer::instance()->Render_Stats.cubemap_update)
	{
		return;
	}

	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	vkCmdDispatch(cmd_buffer, 9, 1, 1);

	m_update = false;
}

void CubemapSHAdd::onImGui()
{
	if (ImGui::Button("Update"))
	{
		m_update = true;
	}

	const auto &IrradianceSH = Renderer::instance()->getRenderGraph()->getAttachment("IrradianceSH");
	ImGui::Text("IrradianceSH Result: ");
	ImGui::Image(ImGuiContext::textureID(IrradianceSH.getView(), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(300, 50));
}
}        // namespace Ilum::pass