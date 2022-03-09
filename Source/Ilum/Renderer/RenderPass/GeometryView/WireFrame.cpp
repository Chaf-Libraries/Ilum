#include "WireFrame.hpp"

#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"

#include "Renderer/Renderer.hpp"

#include <imgui.h>

namespace Ilum::pass
{
void WireFramePass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/GeometryView/WireFrame.vert", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/GeometryView/WireFrame.frag", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::GLSL);

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR,
	    VK_DYNAMIC_STATE_LINE_WIDTH};

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

	state.rasterization_state.cull_mode = VK_CULL_MODE_NONE;

	// Disable blending
	for (auto &color_blend_attachment_state : state.color_blend_attachment_states)
	{
		color_blend_attachment_state.blend_enable = false;
	}

	state.rasterization_state.polygon_mode     = VK_POLYGON_MODE_LINE;
	state.depth_stencil_state.depth_compare_op = VK_COMPARE_OP_LESS_OR_EQUAL;

	state.descriptor_bindings.bind(0, 0, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	state.descriptor_bindings.bind(0, 1, "PerInstanceBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

	state.declareAttachment("GeometryView", VK_FORMAT_R8G8B8A8_UNORM, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("GeometryDepthStencil", VK_FORMAT_D32_SFLOAT_S8_UINT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);

	state.addOutputAttachment("GeometryView", AttachmentState::Load_Color);
	state.addOutputAttachment("GeometryDepthStencil", AttachmentState::Load_Depth_Stencil);
}

void WireFramePass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
	resolve.resolve("PerInstanceBuffer", Renderer::instance()->Render_Buffer.Instance_Buffer);
}

void WireFramePass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	const auto surface_group      = Scene::instance()->getRegistry().group<cmpt::SurfaceRenderer>(entt::get<cmpt::Transform, cmpt::Tag>);
	const auto static_mesh_group  = Scene::instance()->getRegistry().group<cmpt::StaticMeshRenderer>(entt::get<cmpt::Transform, cmpt::Tag>);
	const auto dynamic_mesh_group = Scene::instance()->getRegistry().group<cmpt::DynamicMeshRenderer>(entt::get<cmpt::Transform, cmpt::Tag>);

	if ((surface_group.empty() && static_mesh_group.empty() && dynamic_mesh_group.empty()) || !m_enable)
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

	auto &extent = Renderer::instance()->getRenderTargetExtent();

	VkViewport viewport = {0, static_cast<float>(extent.height), static_cast<float>(extent.width), -static_cast<float>(extent.height), 0, 1};
	VkRect2D   scissor  = {0, 0, extent.width, extent.height};

	vkCmdSetViewport(cmd_buffer, 0, 1, &viewport);
	vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);
	vkCmdSetLineWidth(cmd_buffer, m_line_width);

	// Draw surface wire frame
	surface_group.each([&cmd_buffer, state](const entt::entity &entity, const cmpt::SurfaceRenderer &surface_renderer, const cmpt::Transform &transform, const cmpt::Tag &tag) {
		if (surface_renderer.vertex_buffer)
		{
			VkDeviceSize offsets[1] = {0};
			vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &surface_renderer.vertex_buffer.getBuffer(), offsets);
			vkCmdBindIndexBuffer(cmd_buffer, surface_renderer.index_buffer.getBuffer(), 0, VK_INDEX_TYPE_UINT32);

			struct
			{
				glm::mat4 transform;
				uint32_t  dynamic;
			} push_block;

			push_block.transform = transform.world_transform;
			push_block.dynamic   = true;

			vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_block), &push_block);
			vkCmdDrawIndexed(cmd_buffer, static_cast<uint32_t>(surface_renderer.index_buffer.getSize()) / sizeof(uint32_t), 1, 0, 0, 0);
		}
	});

	// Draw static mesh
	{
		const auto &vertex_buffer = Renderer::instance()->Render_Buffer.Static_Vertex_Buffer;
		const auto &index_buffer  = Renderer::instance()->Render_Buffer.Static_Index_Buffer;

		if (Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count > 0 && vertex_buffer.getBuffer() && index_buffer.getBuffer())
		{
			VkDeviceSize offsets[1] = {0};
			vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &vertex_buffer.getBuffer(), offsets);
			vkCmdBindIndexBuffer(cmd_buffer, index_buffer.getBuffer(), 0, VK_INDEX_TYPE_UINT32);

			struct
			{
				glm::mat4 transform;
				uint32_t  dynamic;
			} push_block;

			push_block.dynamic = false;

			vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_block), &push_block);

			auto &draw_buffer  = Renderer::instance()->Render_Buffer.Command_Buffer;
			auto &count_buffer = Renderer::instance()->Render_Buffer.Count_Buffer;

			vkCmdDrawIndexedIndirectCount(cmd_buffer, draw_buffer, 0, count_buffer, 0, Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count, sizeof(VkDrawIndexedIndirectCommand));
		}
	}

	// Draw dynamic mesh
	{
		if (!dynamic_mesh_group.empty())
		{
			uint32_t instance_id = Renderer::instance()->Render_Stats.static_mesh_count.instance_count;
			dynamic_mesh_group.each([this, &cmd_buffer, &instance_id, state](const entt::entity &entity, const cmpt::DynamicMeshRenderer &mesh_renderer, const cmpt::Transform &transform, const cmpt::Tag &tag) {
				if (mesh_renderer.vertex_buffer && mesh_renderer.index_buffer)
				{
					VkDeviceSize offsets[1] = {0};
					vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &mesh_renderer.vertex_buffer.getBuffer(), offsets);
					vkCmdBindIndexBuffer(cmd_buffer, mesh_renderer.index_buffer, 0, VK_INDEX_TYPE_UINT32);

					struct
					{
						glm::mat4 transform;
						uint32_t  dynamic;
					} push_block;

					push_block.dynamic = false;

					vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_block), &push_block);
					vkCmdDrawIndexed(cmd_buffer, static_cast<uint32_t>(mesh_renderer.indices.size()), 1, 0, 0, instance_id++);
				}
			});
		}
	}

	vkCmdEndRenderPass(cmd_buffer);
}

void WireFramePass::onImGui()
{
	ImGui::Checkbox("Enable", &m_enable);
	ImGui::DragFloat("Width", &m_line_width, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
}
}        // namespace Ilum::pass