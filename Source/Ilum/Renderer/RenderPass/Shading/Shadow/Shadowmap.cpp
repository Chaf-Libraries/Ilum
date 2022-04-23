#include "Shadowmap.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Scene/Component/Camera.hpp"
#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Device/LogicalDevice.hpp"

#include "ImGui/ImGuiContext.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>

namespace Ilum::pass
{
void ShadowmapPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Shadow/Shadowmap.hlsl", VK_SHADER_STAGE_TASK_BIT_NV, Shader::Type::HLSL, "ASmain");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Shadow/Shadowmap.hlsl", VK_SHADER_STAGE_MESH_BIT_NV, Shader::Type::HLSL, "MSmain");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Shadow/Shadowmap.hlsl", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::HLSL, "PSmain");

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR,
	    VK_DYNAMIC_STATE_DEPTH_BIAS};

	state.color_blend_attachment_states.resize(1);
	state.depth_stencil_state.stencil_test_enable = false;

	state.descriptor_bindings.bind(0, 0, "PerInstanceBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 1, "PerMeshletBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 2, "Vertices", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 3, "MeshletVertexBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 4, "MeshletIndexBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 5, "SpotLights", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 6, "CullingBuffer", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

	state.declareAttachment("Shadowmap", VK_FORMAT_D32_SFLOAT, m_resolution.width, m_resolution.height, false, static_cast<uint32_t>(Renderer::instance()->Render_Buffer.Spot_Light_Buffer.getSize()) / sizeof(cmpt::SpotLight));
	state.addOutputAttachment("Shadowmap", VkClearDepthStencilValue{1.f, 0u});
}

void ShadowmapPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("PerInstanceBuffer", Renderer::instance()->Render_Buffer.Instance_Buffer);
	resolve.resolve("PerMeshletBuffer", Renderer::instance()->Render_Buffer.Meshlet_Buffer);
	resolve.resolve("Vertices", Renderer::instance()->Render_Buffer.Static_Vertex_Buffer);
	resolve.resolve("MeshletVertexBuffer", Renderer::instance()->Render_Buffer.Meshlet_Vertex_Buffer);
	resolve.resolve("MeshletIndexBuffer", Renderer::instance()->Render_Buffer.Meshlet_Index_Buffer);
	resolve.resolve("CullingBuffer", Renderer::instance()->Render_Buffer.Culling_Buffer);
	resolve.resolve("SpotLights", Renderer::instance()->Render_Buffer.Spot_Light_Buffer);
}

void ShadowmapPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	VkRenderPassBeginInfo begin_info = {};
	begin_info.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	begin_info.renderPass            = state.pass.render_pass;
	begin_info.renderArea            = state.pass.render_area;
	begin_info.framebuffer           = state.pass.frame_buffer;
	begin_info.clearValueCount       = static_cast<uint32_t>(state.pass.clear_values.size());
	begin_info.pClearValues          = state.pass.clear_values.data();

	vkCmdBeginRenderPass(cmd_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	VkViewport viewport = {0, static_cast<float>(m_resolution.height), static_cast<float>(m_resolution.width), -static_cast<float>(m_resolution.height), 0, 1};
	VkRect2D   scissor  = {0, 0, m_resolution.width, m_resolution.height};

	vkCmdSetViewport(cmd_buffer, 0, 1, &viewport);
	vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);

	vkCmdSetDepthBias(
	    cmd_buffer,
	    m_depth_bias_constant,
	    0.0f,
	    m_depth_bias_slope);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	const auto &spot_lights = Scene::instance()->getRegistry().group<cmpt::SpotLight>(entt::get<cmpt::Transform, cmpt::Tag>);

	for (uint32_t light = 0; light < spot_lights.size(); light++)
	{
		auto spot_light = Entity(spot_lights[light]);

		if (spot_light.getComponent<cmpt::SpotLight>().shadow_mode == 0)
		{
			continue;
		}

		m_push_block.dynamic = 0;
		m_push_block.layer   = light;
		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_TASK_BIT_NV | VK_SHADER_STAGE_MESH_BIT_NV, 0, sizeof(m_push_block), &m_push_block);
		vkCmdDrawMeshTasksNV(cmd_buffer, (Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count + 32 - 1) / 32, 0);
	}

	vkCmdEndRenderPass(cmd_buffer);
}

void ShadowmapPass::onImGui()
{
	ImGui::DragFloat("Depth Bias Constant", &m_depth_bias_constant, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
	ImGui::DragFloat("Depth Bias Slope", &m_depth_bias_slope, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");

	const auto &shadowmap = Renderer::instance()->getRenderGraph()->getAttachment("Shadowmap");

	std::string items;
	for (size_t i = 0; i < shadowmap.getLayerCount(); i++)
	{
		items += std::to_string(i) + '\0';
	}
	items += '\0';
	ImGui::Text("Shadow map: ");
	ImGui::SameLine();
	ImGui::PushItemWidth(100.f);
	ImGui::Combo("Spot Light Index", &m_current_layer, items.data());
	ImGui::PopItemWidth();
	ImGui::Image(ImGuiContext::textureID(shadowmap.getView(m_current_layer), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(100, 100));
}
}        // namespace Ilum::pass