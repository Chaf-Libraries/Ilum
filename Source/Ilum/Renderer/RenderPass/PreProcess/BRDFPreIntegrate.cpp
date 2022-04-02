#include "BRDFPreIntegrate.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "ImGui/ImGuiContext.hpp"

#include <imgui.h>

namespace Ilum::pass
{
void BRDFPreIntegrate::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/BRDFPreIntegrate.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL);

	state.declareAttachment("BRDFPreIntegrate", VK_FORMAT_R16G16_SFLOAT, 512, 512, false, 1);
	state.addOutputAttachment("BRDFPreIntegrate", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 0, "BRDFPreIntegrate", ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void BRDFPreIntegrate::resolveResources(ResolveState &resolve)
{
}

void BRDFPreIntegrate::render(RenderPassState &state)
{
	if (!m_finish)
	{
		return;
	}

	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	vkCmdDispatch(cmd_buffer, 512 / 8, 512 / 8, 1);

	m_finish = false;
}

void BRDFPreIntegrate::onImGui()
{
	const auto &brdf = Renderer::instance()->getRenderGraph()->getAttachment("BRDFPreIntegrate");

	ImGui::Text("BRDF PreIntegrate Result: ");
	ImGui::Image(ImGuiContext::textureID(brdf.getView(), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(100, 100));
}
}        // namespace Ilum::pass