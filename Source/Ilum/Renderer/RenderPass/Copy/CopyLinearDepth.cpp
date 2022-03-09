#pragma once

#include "CopyLinearDepth.hpp"

#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Device/PhysicalDevice.hpp"

#include <imgui.h>

namespace Ilum::pass
{
void CopyLinearDepth::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Copy/CopyLinearDepth.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	// GBuffer 2: R - Metallic, G - Roughness, B - Subsurface, A - EntityID
	state.descriptor_bindings.bind(0, 0, "GBuffer1", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 1, "LinearDepth", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void CopyLinearDepth::resolveResources(ResolveState &resolve)
{

}

void CopyLinearDepth::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkExtent2D), &Renderer::instance()->getRenderTargetExtent());

	vkCmdDispatch(cmd_buffer, (Renderer::instance()->getRenderTargetExtent().width + 32 - 1) / 32, (Renderer::instance()->getRenderTargetExtent().height + 32 - 1) / 32, 1);
}
}        // namespace Ilum::pass