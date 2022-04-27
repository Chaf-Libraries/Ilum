#include "TAA.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>

#define HALTION_SAMPLES 16

namespace Ilum::pass
{
// Camera jitter
inline float halton_sequence(uint32_t base, uint32_t index)
{
	float result = 0.f;
	float f      = 1.f;

	while (index > 0)
	{
		f /= static_cast<float>(base);
		result += f * (index % base);
		index = static_cast<uint32_t>(floorf(static_cast<float>(index) / static_cast<float>(base)));
	}

	return result;
}

TAAPass::TAAPass(const std::string &input, const std::string &prev, const std::string &output) :
    m_input(input), m_prev(prev), m_output(output)
{
}

void TAAPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PostProcess/TAA.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL, "main");

	state.color_blend_attachment_states[0].blend_enable = false;

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR};

	state.declareAttachment(m_output, VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.addOutputAttachment(m_output, AttachmentState::Clear_Color);

	state.declareAttachment(m_prev, VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.addOutputAttachment(m_prev, AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 0, m_prev, Renderer::instance()->getSampler(Renderer::SamplerType::Bilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 1, m_input, Renderer::instance()->getSampler(Renderer::SamplerType::Point_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 2, "GBuffer1", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 3, "GBuffer3", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 4, m_output, ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	state.descriptor_bindings.bind(0, 5, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
}

void TAAPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
}

void TAAPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	auto &extent = Renderer::instance()->getRenderTargetExtent();

	vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_push_constants), &m_push_constants);
	vkCmdDispatch(cmd_buffer, (extent.width + 32 - 1) / 32, (extent.height + 32 - 1) / 32, 1);
}

void TAAPass::onImGui()
{
	ImGui::Checkbox("Sharpen", reinterpret_cast<bool *>(&m_push_constants.sharpen));
	ImGui::SliderFloat("Feedback Min", &m_push_constants.feedback_min, 0.f, 1.f, "%.3f");
	ImGui::SliderFloat("Feedback Max", &m_push_constants.feedback_max, m_push_constants.feedback_min, 1.f, "%.3f");
}
}        // namespace Ilum::pass