#pragma once

#include "MeshletCullingPass.hpp"

#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Device/PhysicalDevice.hpp"

namespace Ilum::pass
{
MeshletCullingPass::MeshletCullingPass()
{
}

void MeshletCullingPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/meshlet_culling.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 0, "IndirectDrawCommand", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 1, "PerInstanceData", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 2, "PerMeshletData", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 3, "DrawInfo", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 4, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	state.descriptor_bindings.bind(0, 5, "hiz - buffer", Renderer::instance()->getSampler(Renderer::SamplerType::Point_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 6, "count_buffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 7, "culling_buffer", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	state.descriptor_bindings.bind(0, 8, "InstanceVisibility", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

void MeshletCullingPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("IndirectDrawCommand", Renderer::instance()->Render_Queue.Command_Buffer);
	resolve.resolve("PerInstanceData", Renderer::instance()->Render_Queue.Instance_Buffer);
	resolve.resolve("PerMeshletData", Renderer::instance()->Render_Queue.Meshlet_Buffer);
	resolve.resolve("DrawInfo", Renderer::instance()->Render_Queue.Draw_Buffer);
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
	resolve.resolve("hiz - buffer", *Renderer::instance()->Last_Frame.hiz_buffer);
	resolve.resolve("count_buffer", Renderer::instance()->Render_Queue.Count_Buffer);
	resolve.resolve("culling_buffer", Renderer::instance()->Render_Queue.Culling_Buffer);
	resolve.resolve("InstanceVisibility", Renderer::instance()->Render_Queue.Instance_Visibility_Buffer);
}

void MeshletCullingPass::render(RenderPassState &state)
{
	{
		std::memcpy(&Renderer::instance()->Meshlet_Visible, Renderer::instance()->Render_Queue.Count_Buffer.map(), sizeof(uint32_t));
		std::memcpy(&Renderer::instance()->Instance_Visible, reinterpret_cast<uint32_t *>(Renderer::instance()->Render_Queue.Count_Buffer.map()) + 1, sizeof(uint32_t));
		Renderer::instance()->Render_Queue.Count_Buffer.unmap();
	}


	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	vkCmdDispatch(cmd_buffer, (Renderer::instance()->Meshlet_Count + 64 - 1) / 64, 1, 1);
}
}        // namespace Ilum::pass