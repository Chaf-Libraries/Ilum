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

TAAPass::TAAPass()
{
	for (uint32_t i = 1; i <= HALTION_SAMPLES; i++)
	{
		m_jitter_samples.push_back(glm::vec2(2.f * halton_sequence(2, i) - 1.f, 2.f * halton_sequence(3, i) - 1.f));
	}
}

void TAAPass::onUpdate()
{
	if (m_enable)
	{
		// Jitter camera
		CameraData *camera_data = reinterpret_cast<CameraData *>(Renderer::instance()->Render_Buffer.Camera_Buffer.map());

		m_prev_jitter = m_current_jitter;

		uint32_t  sample_idx = static_cast<uint32_t>(GraphicsContext::instance()->getFrameCount() % m_jitter_samples.size());
		glm::vec2 halton     = m_jitter_samples[sample_idx];

		auto rt_extent = Renderer::instance()->getRenderTargetExtent();

		m_current_jitter = glm::vec2(halton.x / static_cast<float>(rt_extent.width), halton.y / static_cast<float>(rt_extent.height));
	}
}

void TAAPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/PostProcess/TAA.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.color_blend_attachment_states[0].blend_enable = false;

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR};

	state.descriptor_bindings.bind(0, 0, "last_result", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 1, "lighting", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 2, "gbuffer - motion_vector_curvature", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 3, "gbuffer - linear_depth", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	state.declareAttachment("taa_result", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.addOutputAttachment("taa_result", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 4, "taa_result", ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void TAAPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("last_result", *Renderer::instance()->Last_Frame.last_result);
}

void TAAPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	if (m_enable)
	{
		vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

		for (auto &descriptor_set : state.pass.descriptor_sets)
		{
			vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
		}

		auto &rt_extent = Renderer::instance()->getRenderTargetExtent();

		glm::vec4 jitter = glm::vec4(m_current_jitter, m_prev_jitter);

		glm::vec4 extent = glm::vec4(1.f / static_cast<float>(rt_extent.width), 1.f / static_cast<float>(rt_extent.height), static_cast<float>(rt_extent.width), static_cast<float>(rt_extent.height));
		uint32_t  enable = static_cast<uint32_t>(m_enable);

		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(glm::vec4), glm::value_ptr(jitter));
		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, sizeof(glm::vec4), sizeof(glm::vec4), glm::value_ptr(extent));
		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 2 * sizeof(glm::vec4), sizeof(glm::vec2), glm::value_ptr(m_feedback));
		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 2 * sizeof(glm::vec4) + sizeof(glm::vec2), sizeof(uint32_t), &m_sharpen);

		vkCmdDispatch(cmd_buffer, (rt_extent.width + 32 - 1) / 32, (rt_extent.height + 32 - 1) / 32, 1);
	}
	else
	{
		cmd_buffer.copyImage(
		    ImageInfo{state.graph.getAttachment("lighting"), VK_IMAGE_USAGE_SAMPLED_BIT},
		    ImageInfo{state.graph.getAttachment("taa_result"), VK_IMAGE_USAGE_STORAGE_BIT});
		cmd_buffer.transferLayout(state.graph.getAttachment("lighting"), VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_USAGE_SAMPLED_BIT);
		cmd_buffer.transferLayout(state.graph.getAttachment("taa_result"), VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_USAGE_STORAGE_BIT);
	}
}

void TAAPass::onImGui()
{
	ImGui::Checkbox("Enable", &m_enable);
	ImGui::Checkbox("Sharpen", reinterpret_cast<bool*>(&m_sharpen));
	ImGui::SliderFloat("Feedback Min", &m_feedback.x, 0.f, 1.f, "%.3f");
	ImGui::SliderFloat("Feedback Max", &m_feedback.y, m_feedback.x, 1.f, "%.3f");
}
}        // namespace Ilum::pass