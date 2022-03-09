#include "BloomBlur.hpp"

#include "Renderer/Renderer.hpp"
#include "Renderer/RenderGraph/RenderGraph.hpp"

#include <imgui.h>

namespace Ilum::pass
{
BloomBlur::BloomBlur(const std::string &input, const std::string &output, bool horizontal) :
    m_input(input), m_output(output)
{
	m_push_data.horizontal = horizontal;
}

void BloomBlur::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PostProcess/BloomBlur.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 0, m_input, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	
	state.declareAttachment(m_output, VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.addOutputAttachment(m_output, AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 1, m_output, ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void BloomBlur::resolveResources(ResolveState &resolve)
{
}

void BloomBlur::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	if (m_enable)
	{
		vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

		for (auto &descriptor_set : state.pass.descriptor_sets)
		{
			vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
		}

		m_push_data.extent = Renderer::instance()->getRenderTargetExtent();

		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_push_data), &m_push_data);

		vkCmdDispatch(cmd_buffer, (m_push_data.extent.width + 32 - 1) / 32, (m_push_data.extent.height + 32 - 1) / 32, 1);
	}
	else
	{
		VkClearColorValue clear_color = {};
		clear_color.float32[0]        = 0.f;
		clear_color.float32[1]        = 0.f;
		clear_color.float32[2]        = 0.f;
		clear_color.float32[3]        = 0.f;
		vkCmdClearColorImage(cmd_buffer, state.graph.getAttachment(m_output), VK_IMAGE_LAYOUT_GENERAL, &clear_color, 1, &state.graph.getAttachment(m_output).getSubresourceRange());
	}
}

void BloomBlur::onImGui()
{
	ImGui::Checkbox("Enable", reinterpret_cast<bool *>(&m_enable));
	ImGui::Checkbox("Horizental", reinterpret_cast<bool *>(&m_push_data.horizontal));
	ImGui::DragFloat("Scale", &m_push_data.scale, 0.001f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	ImGui::DragFloat("Strength", &m_push_data.strength, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
}
}        // namespace Ilum::pass