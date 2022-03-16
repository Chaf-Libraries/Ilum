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
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Shadow/OmniShadowmap.vert", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Shadow/OmniShadowmap.frag", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::GLSL);

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR};

	state.vertex_input_state.attribute_descriptions = {
	    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
	    VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texcoord)},
	    VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
	    VkVertexInputAttributeDescription{3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent)},
	    VkVertexInputAttributeDescription{4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, bitangent)}};

	state.vertex_input_state.binding_descriptions = {
	    VkVertexInputBindingDescription{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};

	state.color_blend_attachment_states.resize(1);
	state.depth_stencil_state.stencil_test_enable = false;

	// Disable blending
	for (auto &color_blend_attachment_state : state.color_blend_attachment_states)
	{
		color_blend_attachment_state.blend_enable = false;
	}

	state.descriptor_bindings.bind(0, 0, "PerInstanceBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 1, "PointLights", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

	state.declareAttachment("OmniShadowmap", VK_FORMAT_D32_SFLOAT, m_resolution.width, m_resolution.height, false, static_cast<uint32_t>(Renderer::instance()->Render_Buffer.Point_Light_Buffer.getSize()) / sizeof(cmpt::PointLight) * 6);
	state.addOutputAttachment("OmniShadowmap", VkClearDepthStencilValue{1.f, 0u});
}

void OmniShadowmapPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("PerInstanceBuffer", Renderer::instance()->Render_Buffer.Instance_Buffer);
	resolve.resolve("PointLights", Renderer::instance()->Render_Buffer.Point_Light_Buffer);
}

void OmniShadowmapPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	uint32_t point_light_count = Renderer::instance()->Render_Stats.light_count.point_light_count;

	auto &camera_entity = Renderer::instance()->Main_Camera;

	if (!camera_entity || (!camera_entity.hasComponent<cmpt::PerspectiveCamera>() && !camera_entity.hasComponent<cmpt::OrthographicCamera>()))
	{
		return;
	}

	if (point_light_count == 0)
	{
		return;
	}

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	VkViewport viewport = {0, static_cast<float>(m_resolution.height), static_cast<float>(m_resolution.width), -static_cast<float>(m_resolution.height), 0, 1};
	VkRect2D   scissor  = {0, 0, m_resolution.width, m_resolution.height};

	vkCmdSetViewport(cmd_buffer, 0, 1, &viewport);
	vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);

	VkRenderPassBeginInfo begin_info = {};
	begin_info.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	begin_info.renderPass            = state.pass.render_pass;
	begin_info.renderArea            = state.pass.render_area;
	begin_info.framebuffer           = state.pass.frame_buffer;
	begin_info.clearValueCount       = static_cast<uint32_t>(state.pass.clear_values.size());
	begin_info.pClearValues          = state.pass.clear_values.data();

	vkCmdBeginRenderPass(cmd_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

	const auto &point_lights = Scene::instance()->getRegistry().group<cmpt::PointLight>(entt::get<cmpt::Transform, cmpt::Tag>);

	for (uint32_t light = 0; light < point_lights.size(); light++)
	{
		auto point_light = Entity(point_lights[light]);

		if (point_light.getComponent<cmpt::PointLight>().shadow_mode == 0)
		{
			continue;
		}

		// Draw static mesh
		{
			const auto &vertex_buffer = Renderer::instance()->Render_Buffer.Static_Vertex_Buffer;
			const auto &index_buffer  = Renderer::instance()->Render_Buffer.Static_Index_Buffer;

			if (Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count > 0 && vertex_buffer.getBuffer() && index_buffer.getBuffer())
			{
				VkDeviceSize offsets[1] = {0};
				vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &vertex_buffer.getBuffer(), offsets);
				vkCmdBindIndexBuffer(cmd_buffer, index_buffer.getBuffer(), 0, VK_INDEX_TYPE_UINT32);

				auto &draw_buffer  = Renderer::instance()->Render_Buffer.Command_Buffer;
				auto &count_buffer = Renderer::instance()->Render_Buffer.Count_Buffer;

				m_push_block.dynamic   = 0;
				m_push_block.light_id  = light;
				m_push_block.light_pos = point_light.getComponent<cmpt::PointLight>().position;

				for (uint32_t i = 0; i < 6; i++)
				{
					m_push_block.face_id         = i;
					m_push_block.view_projection = get_point_light_shadow_matrix(point_light.getComponent<cmpt::PointLight>().position, i);
					vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(m_push_block), &m_push_block);
					vkCmdDrawIndexedIndirectCount(cmd_buffer, draw_buffer, 0, count_buffer, sizeof(uint32_t), Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count, sizeof(VkDrawIndexedIndirectCommand));
				}
			}
		}

		// Draw dynamic mesh
		{
			Renderer::instance()->Render_Stats.dynamic_mesh_count.instance_count = 0;
			Renderer::instance()->Render_Stats.dynamic_mesh_count.triangle_count = 0;

			const auto group = Scene::instance()->getRegistry().group<cmpt::DynamicMeshRenderer>(entt::get<cmpt::Transform, cmpt::Tag>);

			if (!group.empty())
			{
				uint32_t instance_id = Renderer::instance()->Render_Stats.static_mesh_count.instance_count;
				group.each([&](const entt::entity &entity, const cmpt::DynamicMeshRenderer &mesh_renderer, const cmpt::Transform &transform, const cmpt::Tag &tag) {
					if (mesh_renderer.vertex_buffer && mesh_renderer.index_buffer)
					{
						Renderer::instance()->Render_Stats.dynamic_mesh_count.instance_count++;
						Renderer::instance()->Render_Stats.dynamic_mesh_count.triangle_count += static_cast<uint32_t>(mesh_renderer.indices.size()) / 3;

						VkDeviceSize offsets[1] = {0};
						vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &mesh_renderer.vertex_buffer.getBuffer(), offsets);
						vkCmdBindIndexBuffer(cmd_buffer, mesh_renderer.index_buffer, 0, VK_INDEX_TYPE_UINT32);

						m_push_block.dynamic   = 1;
						m_push_block.light_id  = light;
						m_push_block.transform = transform.world_transform;
						m_push_block.light_pos = point_light.getComponent<cmpt::PointLight>().position;

						for (uint32_t i = 0; i < 6; i++)
						{
							m_push_block.face_id         = i;
							m_push_block.view_projection = get_point_light_shadow_matrix(point_light.getComponent<cmpt::PointLight>().position, i);
							vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(m_push_block), &m_push_block);
							vkCmdDrawIndexed(cmd_buffer, static_cast<uint32_t>(mesh_renderer.index_buffer.getSize() / sizeof(uint32_t)), 1, 0, 0, instance_id++);
						}
					}
				});
			}
		}
	}

	vkCmdEndRenderPass(cmd_buffer);
}

void OmniShadowmapPass::onImGui()
{
	ImGui::DragFloat("Depth Bias", &m_push_block.depth_bias, 0.001f, 0.f, std::numeric_limits<float>::max(), "%.3f");

	const auto &shadowmap = Renderer::instance()->getRenderGraph()->getAttachment("CascadeShadowmap");

	std::string light_id = "";
	for (size_t i = 0; i < shadowmap.getLayerCount() / 4; i++)
	{
		light_id += std::to_string(i) + '\0';
	}
	light_id += '\0';

	std::string face_id = "+X\0-X\0+Y\0-Y\0+Z\0-Z\0";

	ImGui::Text("Cascade Shadowmap: ");
	ImGui::PushItemWidth(100.f);
	ImGui::Combo("Directional Light Index", &m_light_id, light_id.data());
	ImGui::Combo("Directional Cascade Index", &m_face_id, face_id.data());
	ImGui::PopItemWidth();
	ImGui::Image(ImGuiContext::textureID(shadowmap.getView(m_light_id * 4 + m_face_id), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(100, 100));
}
}        // namespace Ilum::pass