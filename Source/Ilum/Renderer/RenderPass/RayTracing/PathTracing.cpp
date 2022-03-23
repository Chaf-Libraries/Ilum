#include "PathTracing.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Graphics/Vulkan/VK_Debugger.h"

#include "ImGui/ImGuiContext.hpp"

#include <imgui.h>

namespace Ilum::pass
{
void PathTracing::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/PathTracing.rgen", VK_SHADER_STAGE_RAYGEN_BIT_KHR, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/PathTracing.rchit", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/PathTracing.rmiss", VK_SHADER_STAGE_MISS_BIT_KHR, Shader::Type::GLSL);

	state.declareAttachment("PathTracing", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.addOutputAttachment("PathTracing", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 0, "PathTracing", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void PathTracing::resolveResources(ResolveState &resolve)
{
}

void PathTracing::render(RenderPassState &state)
{
	if (!m_finish)
	{
		auto &cmd_buffer = state.command_buffer;

		vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

		for (auto &descriptor_set : state.pass.descriptor_sets)
		{
			vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
		}

		vkCmdDispatch(cmd_buffer, 1024 / 32, 1024 / 32, 1);

		m_finish = true;
	}
}

void PathTracing::onImGui()
{

}
}        // namespace Ilum::pass