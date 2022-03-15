#include "CascadeShadowmap.hpp"

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
void CascadeShadowmapPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Shadow/CascadeShadowmap.vert", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::GLSL);

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR,
	    VK_DYNAMIC_STATE_DEPTH_BIAS};

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
	state.descriptor_bindings.bind(0, 1, "DirectionalLights", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

	state.declareAttachment("CascadeShadowmap", VK_FORMAT_D32_SFLOAT, m_resolution.width, m_resolution.height, false, static_cast<uint32_t>(Renderer::instance()->Render_Buffer.Spot_Light_Buffer.getSize()) / sizeof(cmpt::SpotLight) * 4);
	state.addOutputAttachment("CascadeShadowmap", VkClearDepthStencilValue{1.f, 0u});
}

void CascadeShadowmapPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("PerInstanceBuffer", Renderer::instance()->Render_Buffer.Instance_Buffer);
	resolve.resolve("DirectionalLights", Renderer::instance()->Render_Buffer.Directional_Light_Buffer);
}

void CascadeShadowmapPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	uint32_t directional_light_count = Renderer::instance()->Render_Stats.light_count.directional_light_count;

	auto &camera_entity = Renderer::instance()->Main_Camera;

	if (!camera_entity || (!camera_entity.hasComponent<cmpt::PerspectiveCamera>() && !camera_entity.hasComponent<cmpt::OrthographicCamera>()))
	{
		return;
	}

	if (directional_light_count == 0)
	{
		return;
	}

	VkRenderPassBeginInfo begin_info = {};
	begin_info.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	begin_info.renderPass            = state.pass.render_pass;
	begin_info.renderArea            = state.pass.render_area;
	begin_info.framebuffer           = state.pass.frame_buffer;
	begin_info.clearValueCount       = static_cast<uint32_t>(state.pass.clear_values.size());
	begin_info.pClearValues          = state.pass.clear_values.data();

	vkCmdBeginRenderPass(cmd_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	VkViewport viewport = {0, static_cast<float>(m_resolution.height), static_cast<float>(m_resolution.width), -static_cast<float>(m_resolution.height), 0, 1};
	VkRect2D   scissor  = {0, 0, m_resolution.width, m_resolution.height};

	vkCmdSetViewport(cmd_buffer, 0, 1, &viewport);
	vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);

	vkCmdSetDepthBias(
	    cmd_buffer,
	    m_depth_bias_constant,
	    0.0f,
	    m_depth_bias_slope);

	const auto &directional_lights = Scene::instance()->getRegistry().group<cmpt::DirectionalLight>(entt::get<cmpt::Transform, cmpt::Tag>);

	for (uint32_t light = 0; light < directional_lights.size(); light++)
	{
		auto spot_light = Entity(directional_lights[light]);

		// Draw static mesh
		{
			const auto &vertex_buffer = Renderer::instance()->Render_Buffer.Static_Vertex_Buffer;
			const auto &index_buffer  = Renderer::instance()->Render_Buffer.Static_Index_Buffer;

			if (Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count > 0 && vertex_buffer.getBuffer() && index_buffer.getBuffer())
			{
				VkDeviceSize offsets[1] = {0};
				vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &vertex_buffer.getBuffer(), offsets);
				vkCmdBindIndexBuffer(cmd_buffer, index_buffer.getBuffer(), 0, VK_INDEX_TYPE_UINT32);

				m_push_block.dynamic  = 0;
				m_push_block.light_id = light;

				for (uint32_t i = 0; i < 4; i++)
				{
					m_push_block.cascaded_id = i;

					vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(m_push_block), &m_push_block);

					auto &draw_buffer  = Renderer::instance()->Render_Buffer.Command_Buffer;
					auto &count_buffer = Renderer::instance()->Render_Buffer.Count_Buffer;
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
						m_push_block.transform = transform.world_transform;
						m_push_block.light_id  = light;

						for (uint32_t i = 0; i < 4; i++)
						{
							m_push_block.cascaded_id = i;

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

void CascadeShadowmapPass::onImGui()
{
	ImGui::DragFloat("Depth Bias Constant", &m_depth_bias_constant, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
	ImGui::DragFloat("Depth Bias Slope", &m_depth_bias_slope, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");

	const auto &shadowmap = Renderer::instance()->getRenderGraph()->getAttachment("CascadeShadowmap");

	std::string light_id = "";
	for (size_t i = 0; i < shadowmap.getLayerCount() / 4; i++)
	{
		light_id += std::to_string(i) + '\0';
	}
	light_id += '\0';

	std::string cascade_id = "";
	for (size_t i = 0; i < 4; i++)
	{
		cascade_id += std::to_string(i) + '\0';
	}
	cascade_id += '\0';
	ImGui::Text("Cascade Shadowmap: ");
	ImGui::PushItemWidth(100.f);
	ImGui::Combo("Directional Light Index", &m_light_index, light_id.data());
	ImGui::Combo("Directional Cascade Index", &m_cascade_index, cascade_id.data());
	ImGui::PopItemWidth();
	ImGui::Image(ImGuiContext::textureID(shadowmap.getView(m_light_index * 4 + m_cascade_index), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(100, 100));
}
}        // namespace Ilum::pass