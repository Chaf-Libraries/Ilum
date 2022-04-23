#include "OmniShadowmap.hpp"

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
inline glm::mat4 get_point_light_shadow_matrix(const glm::vec3 &position, uint32_t face)
{
	glm::mat4 projection_matrix = glm::perspective(glm::radians(90.0f), 1.0f, 0.01f, 100.f);

	switch (face)
	{
		case 0:        // POSITIVE_X
			return projection_matrix * glm::lookAt(position, position + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		case 1:        // NEGATIVE_X
			return projection_matrix * glm::lookAt(position, position + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		case 2:        // POSITIVE_Y
			return projection_matrix * glm::lookAt(position, position + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		case 3:        // NEGATIVE_Y
			return projection_matrix * glm::lookAt(position, position + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
		case 4:        // POSITIVE_Z
			return projection_matrix * glm::lookAt(position, position + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		case 5:        // NEGATIVE_Z
			return projection_matrix * glm::lookAt(position, position + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}

	return glm::mat4(1.f);
}

void OmniShadowmapPass::setupPipeline(PipelineState &state)
{
	/*state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/GLSL/Shading/Shadow/OmniShadowmap.vert", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/GLSL/Shading/Shadow/OmniShadowmap.frag", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::GLSL);*/

	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Shadow/OmniShadowmap.hlsl", VK_SHADER_STAGE_TASK_BIT_NV, Shader::Type::HLSL, "ASmain");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Shadow/OmniShadowmap.hlsl", VK_SHADER_STAGE_MESH_BIT_NV, Shader::Type::HLSL, "MSmain");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Shadow/OmniShadowmap.hlsl", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::HLSL, "PSmain");

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR};

	state.color_blend_attachment_states.resize(1);
	state.depth_stencil_state.stencil_test_enable = false;

	// Disable blending
	for (auto &color_blend_attachment_state : state.color_blend_attachment_states)
	{
		color_blend_attachment_state.blend_enable = false;
	}

	state.descriptor_bindings.bind(0, 0, "PerInstanceBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 1, "PerMeshletBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 2, "Vertices", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 3, "MeshletVertexBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 4, "MeshletIndexBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 5, "PointLights", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 6, "CullingBuffer", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

	state.declareAttachment("OmniShadowmap", VK_FORMAT_D32_SFLOAT, m_resolution.width, m_resolution.height, false, static_cast<uint32_t>(Renderer::instance()->Render_Buffer.Point_Light_Buffer.getSize()) / sizeof(cmpt::PointLight) * 6);
	state.addOutputAttachment("OmniShadowmap", VkClearDepthStencilValue{1.f, 0u});
}

void OmniShadowmapPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("PerInstanceBuffer", Renderer::instance()->Render_Buffer.Instance_Buffer);
	resolve.resolve("PerMeshletBuffer", Renderer::instance()->Render_Buffer.Meshlet_Buffer);
	resolve.resolve("Vertices", Renderer::instance()->Render_Buffer.Static_Vertex_Buffer);
	resolve.resolve("MeshletVertexBuffer", Renderer::instance()->Render_Buffer.Meshlet_Vertex_Buffer);
	resolve.resolve("MeshletIndexBuffer", Renderer::instance()->Render_Buffer.Meshlet_Index_Buffer);
	resolve.resolve("CullingBuffer", Renderer::instance()->Render_Buffer.Culling_Buffer);
	resolve.resolve("PointLights", Renderer::instance()->Render_Buffer.Point_Light_Buffer);
}

void OmniShadowmapPass::render(RenderPassState &state)
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

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	const auto &point_lights = Scene::instance()->getRegistry().group<cmpt::PointLight>(entt::get<cmpt::Transform, cmpt::Tag>);

	for (uint32_t light = 0; light < point_lights.size(); light++)
	{
		auto point_light = Entity(point_lights[light]);

		if (point_light.getComponent<cmpt::PointLight>().shadow_mode == 0)
		{
			continue;
		}

		m_push_block.light_id = light;

		for (uint32_t face = 0; face < 6; face++)
		{
			m_push_block.face_id = face;
			m_push_block.view_projection = get_point_light_shadow_matrix(point_light.getComponent<cmpt::PointLight>().position, face);

			vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_TASK_BIT_NV | VK_SHADER_STAGE_MESH_BIT_NV, 0, sizeof(m_push_block), &m_push_block);
			vkCmdDrawMeshTasksNV(cmd_buffer, (Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count + 32 - 1) / 32, 0);
		}
	}

	vkCmdEndRenderPass(cmd_buffer);
}

void OmniShadowmapPass::onImGui()
{
	ImGui::DragFloat("Depth Bias", &m_push_block.depth_bias, 0.001f, 0.f, std::numeric_limits<float>::max(), "%.3f");

	const auto &shadowmap = Renderer::instance()->getRenderGraph()->getAttachment("OmniShadowmap");

	std::string light_id = "";
	for (size_t i = 0; i < shadowmap.getLayerCount() / 4; i++)
	{
		light_id += std::to_string(i) + '\0';
	}
	light_id += '\0';

	ImGui::Text("Omnidirectional Shadowmap: ");
	ImGui::PushItemWidth(100.f);
	ImGui::Combo("Point Light Index", &m_light_id, light_id.data());
	ImGui::Combo("Shadow Cubemap Face", &m_face_id, "+X\0-X\0+Y\0-Y\0+Z\0-Z\0\0");
	ImGui::PopItemWidth();
	ImGui::Image(ImGuiContext::textureID(shadowmap.getView(m_light_id * 4 + m_face_id), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(100, 100));
}
}        // namespace Ilum::pass