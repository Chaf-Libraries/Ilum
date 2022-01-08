#include "CurvePass.hpp"

#include "Renderer/Renderer.hpp"

#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "Threading/ThreadPool.hpp"

#include "File/FileSystem.hpp"

#include "Material/PBR.h"

#include <glm/gtc/type_ptr.hpp>

namespace Ilum::pass
{
void CurvePass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/curve.vert", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/curve.frag", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::GLSL);

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR,
	    VK_DYNAMIC_STATE_LINE_WIDTH};

	state.vertex_input_state.attribute_descriptions = {
	    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
	};

	state.vertex_input_state.binding_descriptions = {
	    VkVertexInputBindingDescription{0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX}};

	state.color_blend_attachment_states.resize(3);
	state.depth_stencil_state.stencil_test_enable = false;

	// Disable blending
	for (auto &color_blend_attachment_state : state.color_blend_attachment_states)
	{
		color_blend_attachment_state.blend_enable = false;
	}

	// Setting rasterization state
	switch (Renderer::instance()->Render_Mode)
	{
		case Renderer::RenderMode::Polygon:
			state.rasterization_state.polygon_mode = VK_POLYGON_MODE_FILL;
			break;
		case Renderer::RenderMode::WireFrame:
			state.rasterization_state.polygon_mode = VK_POLYGON_MODE_LINE;
			break;
		case Renderer::RenderMode::PointCloud:
			state.rasterization_state.polygon_mode = VK_POLYGON_MODE_POINT;
			break;
		default:
			break;
	}

	state.input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;

	state.descriptor_bindings.bind(0, 0, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

	state.declareAttachment("geometry - curve", VK_FORMAT_R8G8B8A8_UNORM, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("debug - instance", VK_FORMAT_R8G8B8A8_UNORM, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("debug - entity", VK_FORMAT_R32_UINT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);

	state.addOutputAttachment("geometry - curve", AttachmentState::Clear_Color);
	state.addOutputAttachment("debug - instance", AttachmentState::Load_Color);
	state.addOutputAttachment("debug - entity", AttachmentState::Load_Color);
}

void CurvePass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
}

void CurvePass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	const auto group = Scene::instance()->getRegistry().group<cmpt::CurveRenderer>(entt::get<cmpt::Transform, cmpt::Tag>);

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

	group.each([&cmd_buffer, &instance_id, state](const entt::entity &entity, const cmpt::CurveRenderer &curve_renderer, const cmpt::Transform &transform, const cmpt::Tag &tag) {
		if (curve_renderer.vertex_buffer)
		{
			Renderer::instance()->Render_Stats.curve_count.instance_count++;
			Renderer::instance()->Render_Stats.curve_count.vertices_count += static_cast<uint32_t>(curve_renderer.vertices.size());

			vkCmdSetLineWidth(cmd_buffer, curve_renderer.line_width);

			VkDeviceSize offsets[1] = {0};
			vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &curve_renderer.vertex_buffer.getBuffer(), offsets);

			struct
			{
				glm::mat4 transform;
				glm::vec4 color;
				uint32_t  entity_id;
				uint32_t  instance_id;
			} push_block;

			push_block.transform   = transform.world_transform;
			push_block.color       = curve_renderer.base_color;
			push_block.entity_id   = static_cast<uint32_t>(entity);
			push_block.instance_id = instance_id++;

			vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_block), &push_block);
			vkCmdDraw(cmd_buffer, static_cast<uint32_t>(curve_renderer.vertices.size()), 1, 0, 0);
		}
	});

	vkCmdEndRenderPass(cmd_buffer);
}
}        // namespace Ilum::pass