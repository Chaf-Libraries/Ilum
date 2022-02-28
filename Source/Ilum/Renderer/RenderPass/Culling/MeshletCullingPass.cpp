#pragma once

#include "MeshletCullingPass.hpp"

#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Device/PhysicalDevice.hpp"

#include <imgui.h>

namespace Ilum::pass
{
void MeshletCullingPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/Culling/MeshletCulling.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

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
	resolve.resolve("IndirectDrawCommand", Renderer::instance()->Render_Buffer.Command_Buffer);
	resolve.resolve("PerInstanceData", Renderer::instance()->Render_Buffer.Instance_Buffer);
	resolve.resolve("PerMeshletData", Renderer::instance()->Render_Buffer.Meshlet_Buffer);
	resolve.resolve("DrawInfo", Renderer::instance()->Render_Buffer.Draw_Buffer);
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
	resolve.resolve("hiz - buffer", *Renderer::instance()->Last_Frame.hiz_buffer);
	resolve.resolve("count_buffer", Renderer::instance()->Render_Buffer.Count_Buffer);
	resolve.resolve("culling_buffer", Renderer::instance()->Render_Buffer.Culling_Buffer);
	resolve.resolve("InstanceVisibility", Renderer::instance()->Render_Buffer.Instance_Visibility_Buffer);
}

void MeshletCullingPass::render(RenderPassState &state)
{
	{
		std::memcpy(&Renderer::instance()->Render_Stats.static_mesh_count.instance_visible, reinterpret_cast<uint32_t *>(Renderer::instance()->Render_Buffer.Count_Buffer.map()) + 2, sizeof(uint32_t));
		std::memcpy(&Renderer::instance()->Render_Stats.static_mesh_count.meshlet_visible, reinterpret_cast<uint32_t *>(Renderer::instance()->Render_Buffer.Count_Buffer.map()) + 1, sizeof(uint32_t));
		Renderer::instance()->Render_Buffer.Count_Buffer.unmap();
	}

	if (enable_culling)
	{
		auto &cmd_buffer = state.command_buffer;

		vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

		for (auto &descriptor_set : state.pass.descriptor_sets)
		{
			vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
		}

		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_culling_mode), &m_culling_mode);

		vkCmdDispatch(cmd_buffer, (Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count + 64 - 1) / 64, 1, 1);
	}
}

void MeshletCullingPass::onImGui()
{
	ImGui::Checkbox("Enable Frustum Culling", reinterpret_cast<bool *>(&m_culling_mode.enable_frustum_culling));
	ImGui::Checkbox("Enable Back Face Cone Culling", reinterpret_cast<bool *>(&m_culling_mode.enable_backface_culling));
	ImGui::Checkbox("Enable Occlusion Culling", reinterpret_cast<bool *>(&m_culling_mode.enable_occlusion_culling));

	enable_culling = m_culling_mode.enable_frustum_culling == 1 ||
	                 m_culling_mode.enable_backface_culling == 1 ||
	                 m_culling_mode.enable_occlusion_culling == 1;
}
}        // namespace Ilum::pass