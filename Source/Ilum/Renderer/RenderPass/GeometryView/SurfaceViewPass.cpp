#include "SurfaceViewPass.hpp"

#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"

#include "Renderer/Renderer.hpp"

namespace Ilum::pass
{
void SurfaceViewPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/GeometryView/SurfaceView.hlsl", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::HLSL, "VSmain");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/GeometryView/SurfaceView.hlsl", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::HLSL, "PSmain");

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR};

	state.vertex_input_state.attribute_descriptions = {
	    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
	    VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texcoord)},
	    VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)}};

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

	state.rasterization_state.polygon_mode = VK_POLYGON_MODE_FILL;

	state.descriptor_bindings.bind(0, 0, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

	state.declareAttachment("GeometryView", VK_FORMAT_R8G8B8A8_UNORM, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("GeometryDepthStencil", VK_FORMAT_D32_SFLOAT_S8_UINT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);

	state.addOutputAttachment("GeometryView", AttachmentState::Load_Color);
	state.addOutputAttachment("GeometryDepthStencil", AttachmentState::Load_Depth_Stencil);
}

void SurfaceViewPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
}

void SurfaceViewPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	const auto group = Scene::instance()->getRegistry().group<cmpt::SurfaceRenderer>(entt::get<cmpt::Transform, cmpt::Tag>);

	if (group.empty())
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

	Renderer::instance()->Render_Stats.curve_count.instance_count = 0;
	Renderer::instance()->Render_Stats.curve_count.vertices_count = 0;

	uint32_t instance_id = Renderer::instance()->Render_Stats.static_mesh_count.instance_count + Renderer::instance()->Render_Stats.dynamic_mesh_count.instance_count;

	group.each([&cmd_buffer, &instance_id, state](const entt::entity &entity, const cmpt::SurfaceRenderer &surface_renderer, const cmpt::Transform &transform, const cmpt::Tag &tag) {
		if (surface_renderer.vertex_buffer)
		{
			Renderer::instance()->Render_Stats.curve_count.instance_count++;
			Renderer::instance()->Render_Stats.curve_count.vertices_count += static_cast<uint32_t>(surface_renderer.vertices.size());

			VkDeviceSize offsets[1] = {0};
			vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &surface_renderer.vertex_buffer.getBuffer(), offsets);
			vkCmdBindIndexBuffer(cmd_buffer, surface_renderer.index_buffer.getBuffer(), 0, VK_INDEX_TYPE_UINT32);

			struct
			{
				glm::mat4 transform;
			} push_block;

			push_block.transform = transform.world_transform;

			vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push_block), &push_block);
			vkCmdDrawIndexed(cmd_buffer, static_cast<uint32_t>(surface_renderer.index_buffer.getSize()) / sizeof(uint32_t), 1, 0, 0, 0);
		}
	});

	vkCmdEndRenderPass(cmd_buffer);
}
}        // namespace Ilum::pass