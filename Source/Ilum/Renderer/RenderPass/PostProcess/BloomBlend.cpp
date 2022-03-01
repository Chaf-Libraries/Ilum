#include "BloomBlend.hpp"

#include "Renderer/Renderer.hpp"

namespace Ilum::pass
{
BloomBlend::BloomBlend(const std::string &input, const std::string &output) :
    m_input(input), m_output(output)
{
}

void BloomBlend::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/PostProcess/BloomBlend.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 0, m_input, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	state.declareAttachment(m_output, VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.addOutputAttachment(m_output, AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 1, m_output, ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void BloomBlend::resolveResources(ResolveState &resolve)
{
}

void BloomBlend::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	auto &extent = Renderer::instance()->getRenderTargetExtent();

	vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(extent), &extent);
	vkCmdDispatch(cmd_buffer, (extent.width + 32 - 1) / 32, (extent.height + 32 - 1) / 32, 1);
}
}        // namespace Ilum::pass