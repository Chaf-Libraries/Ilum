#include "CubemapSHProjection.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "ImGui/ImGuiContext.hpp"

#include <imgui.h>

namespace Ilum::pass
{
CubemapSHProjection::CubemapSHProjection()
{
}

void CubemapSHProjection::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/CubemapSHProjection.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 1, "SkyBox", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Cube, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	state.declareAttachment("SHIntermediate", VK_FORMAT_R16G16B16A16_SFLOAT, 1024, 1024, false, 6);
	state.addOutputAttachment("SHIntermediate", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 0, "SHIntermediate", ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void CubemapSHProjection::resolveResources(ResolveState &resolve)
{
}

void CubemapSHProjection::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	VkExtent2D extent = {1024, 1024};

	vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(extent), &extent);

	vkCmdDispatch(cmd_buffer, 1024 / 8, 1024 / 8, 1);
}

void CubemapSHProjection::onImGui()
{
	const auto &SHIntermediate = Renderer::instance()->getRenderGraph()->getAttachment("SHIntermediate");
	ImGui::Text("SHIntermediate Result: ");

	ImGui::PushItemWidth(100.f);
	ImGui::Combo("Face index", &m_face_id, "+X\0-X\0+Y\0-Y\0+Z\0-Z\0\0");
	ImGui::PopItemWidth();
	ImGui::Image(ImGuiContext::textureID(SHIntermediate.getView(m_face_id), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(100, 100));
}
}        // namespace Ilum::pass