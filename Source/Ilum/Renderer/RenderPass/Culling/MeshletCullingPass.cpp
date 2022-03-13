#pragma once

#include "MeshletCullingPass.hpp"

#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Vulkan/VK_Debugger.h"

#include "Device/LogicalDevice.hpp"

#include "Device/PhysicalDevice.hpp"

#include <imgui.h>

namespace Ilum::pass
{
MeshletCullingPass::MeshletCullingPass()
{
	VkSamplerCreateInfo createInfo = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

	createInfo.magFilter    = VK_FILTER_LINEAR;
	createInfo.minFilter    = VK_FILTER_LINEAR;
	createInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	createInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	createInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	createInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	createInfo.minLod       = 0;
	createInfo.maxLod       = 16.f;

	VkSamplerReductionModeCreateInfo createInfoReduction = {VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT};
	createInfoReduction.reductionMode                    = VK_SAMPLER_REDUCTION_MODE_MAX;
	createInfo.pNext                                     = &createInfoReduction;

	VkSampler hiz_sampler = VK_NULL_HANDLE;
	vkCreateSampler(GraphicsContext::instance()->getLogicalDevice(), &createInfo, 0, &hiz_sampler);

	m_hiz_sampler = Sampler(hiz_sampler);
}

void MeshletCullingPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Culling/MeshletCulling.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 0, "IndirectDrawCommand", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 1, "PerInstanceBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 2, "PerMeshletBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 3, "DrawBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 4, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	state.descriptor_bindings.bind(0, 5, "HizBuffer", m_hiz_sampler, ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 6, "CountBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 7, "CullingBuffer", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	state.descriptor_bindings.bind(0, 8, "InstanceVisibility", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

void MeshletCullingPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("IndirectDrawCommand", Renderer::instance()->Render_Buffer.Command_Buffer);
	resolve.resolve("PerInstanceBuffer", Renderer::instance()->Render_Buffer.Instance_Buffer);
	resolve.resolve("PerMeshletBuffer", Renderer::instance()->Render_Buffer.Meshlet_Buffer);
	resolve.resolve("DrawBuffer", Renderer::instance()->Render_Buffer.Draw_Buffer);
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
	resolve.resolve("CountBuffer", Renderer::instance()->Render_Buffer.Count_Buffer);
	resolve.resolve("CullingBuffer", Renderer::instance()->Render_Buffer.Culling_Buffer);
	resolve.resolve("InstanceVisibility", Renderer::instance()->Render_Buffer.Instance_Visibility_Buffer);
}

void MeshletCullingPass::render(RenderPassState &state)
{
	{
		std::memcpy(&Renderer::instance()->Render_Stats.static_mesh_count.instance_visible, reinterpret_cast<uint32_t *>(Renderer::instance()->Render_Buffer.Count_Buffer.map()) + 3, sizeof(uint32_t));
		std::memcpy(&Renderer::instance()->Render_Stats.static_mesh_count.meshlet_visible, reinterpret_cast<uint32_t *>(Renderer::instance()->Render_Buffer.Count_Buffer.map()) + 2, sizeof(uint32_t));
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