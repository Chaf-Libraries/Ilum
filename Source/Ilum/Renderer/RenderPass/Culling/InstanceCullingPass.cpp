#pragma once

#include "InstanceCullingPass.hpp"

#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Device/PhysicalDevice.hpp"

namespace Ilum::pass
{
InstanceCullingPass::InstanceCullingPass()
{
}

void InstanceCullingPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/Culling/InstanceCulling.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 0, "PerInstanceData", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 1, "InstanceVisibility", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 2, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	state.descriptor_bindings.bind(0, 3, "hiz - buffer", Renderer::instance()->getSampler(Renderer::SamplerType::Point_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 4, "culling_buffer", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	state.descriptor_bindings.bind(0, 5, "count_buffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

void InstanceCullingPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("PerInstanceData", Renderer::instance()->Render_Buffer.Instance_Buffer);
	resolve.resolve("InstanceVisibility", Renderer::instance()->Render_Buffer.Instance_Visibility_Buffer);
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
	resolve.resolve("hiz - buffer", *Renderer::instance()->Last_Frame.hiz_buffer);
	resolve.resolve("culling_buffer", Renderer::instance()->Render_Buffer.Culling_Buffer);
	resolve.resolve("count_buffer", Renderer::instance()->Render_Buffer.Count_Buffer);
}

void InstanceCullingPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	vkCmdDispatch(cmd_buffer, (Renderer::instance()->Render_Stats.static_mesh_count.instance_count + 64 - 1) / 64, 1, 1);
}
}        // namespace Ilum::pass