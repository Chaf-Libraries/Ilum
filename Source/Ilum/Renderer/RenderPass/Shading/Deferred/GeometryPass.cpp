#include "GeometryPass.hpp"

#include "Renderer/Renderer.hpp"

#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "File/FileSystem.hpp"

#include "Threading/ThreadPool.hpp"

#include <glm/gtc/type_ptr.hpp>

namespace Ilum::pass
{
GeometryPass::GeometryPass()
{
}

void GeometryPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Deferred/Geometry.vert", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Deferred/Geometry.frag", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::GLSL);

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

	state.color_blend_attachment_states.resize(6);
	state.depth_stencil_state.stencil_test_enable = false;

	// Disable blending
	for (auto &color_blend_attachment_state : state.color_blend_attachment_states)
	{
		color_blend_attachment_state.blend_enable = false;
	}

	// Setting rasterization state
	switch (m_render_mode)
	{
		case RenderMode::Polygon:
			state.rasterization_state.polygon_mode = VK_POLYGON_MODE_FILL;
			break;
		case RenderMode::WireFrame:
			state.rasterization_state.polygon_mode = VK_POLYGON_MODE_LINE;
			break;
		case RenderMode::PointCloud:
			state.rasterization_state.polygon_mode = VK_POLYGON_MODE_POINT;
			break;
		default:
			break;
	}

	state.descriptor_bindings.bind(0, 0, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	state.descriptor_bindings.bind(0, 1, "TextureArray", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Wrap), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 2, "PerInstanceBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 3, "PerMeshletBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 4, "MaterialBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

	state.addDependency("IndirectDrawCommand", VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
	state.addDependency("meshlet_count", VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

	// GBuffer 0: RGB - Albedo, A - Anisotropic
	// GBuffer 1: RGB - Normal, A - LinearDepth
	// GBuffer 2: R - Metallic, G - Roughness, B - Subsurface, A - EntityID
	// GBuffer 3: R - Sheen, G - Sheen Tint, B - Clearcoat, A - Clearcoat Gloss
	// GBuffer 4: RG - Velocity, B - Specular, A - Specular Tint
	// GBuffer 5: RGB - Emissive, A - Material Type

	state.declareAttachment("GBuffer0", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("GBuffer1", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("GBuffer2", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("GBuffer3", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("GBuffer4", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("GBuffer4", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("GBuffer5", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("DepthStencil", VK_FORMAT_D32_SFLOAT_S8_UINT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);

	VkClearColorValue clear_color = {};
	clear_color.uint32[3]         = static_cast<uint32_t>(entt::null);
	clear_color.float32[3]        = std::numeric_limits<float>::max();

	state.addOutputAttachment("GBuffer0", AttachmentState::Clear_Color);
	state.addOutputAttachment("GBuffer1", clear_color);
	state.addOutputAttachment("GBuffer2", clear_color);
	state.addOutputAttachment("GBuffer3", AttachmentState::Clear_Color);
	state.addOutputAttachment("GBuffer4", AttachmentState::Clear_Color);
	state.addOutputAttachment("GBuffer5", AttachmentState::Clear_Color);

	state.addOutputAttachment("DepthStencil", VkClearDepthStencilValue{1.f, 0u});
}

void GeometryPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
	resolve.resolve("TextureArray", Renderer::instance()->getResourceCache().getImageReferences());
	resolve.resolve("PerInstanceBuffer", Renderer::instance()->Render_Buffer.Instance_Buffer);
	resolve.resolve("MaterialBuffer", Renderer::instance()->Render_Buffer.Material_Buffer);
	resolve.resolve("PerMeshletBuffer", Renderer::instance()->Render_Buffer.Meshlet_Buffer);
}

void GeometryPass::render(RenderPassState &state)
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

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	auto &extent = Renderer::instance()->getRenderTargetExtent();

	VkViewport viewport = {0, static_cast<float>(extent.height), static_cast<float>(extent.width), -static_cast<float>(extent.height), 0, 1};
	VkRect2D   scissor  = {0, 0, extent.width, extent.height};

	vkCmdSetViewport(cmd_buffer, 0, 1, &viewport);
	vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);

	// Draw static mesh
	{
		const auto &vertex_buffer = Renderer::instance()->Render_Buffer.Static_Vertex_Buffer;
		const auto &index_buffer  = Renderer::instance()->Render_Buffer.Static_Index_Buffer;

		if (Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count > 0 && vertex_buffer.getBuffer() && index_buffer.getBuffer())
		{
			VkDeviceSize offsets[1] = {0};
			vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &vertex_buffer.getBuffer(), offsets);
			vkCmdBindIndexBuffer(cmd_buffer, index_buffer.getBuffer(), 0, VK_INDEX_TYPE_UINT32);

			m_vertex_block.dynamic = 0;
			vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(m_vertex_block), &m_vertex_block);

			auto &draw_buffer  = Renderer::instance()->Render_Buffer.Command_Buffer;
			auto &count_buffer = Renderer::instance()->Render_Buffer.Count_Buffer;
			vkCmdDrawIndexedIndirectCount(cmd_buffer, draw_buffer, 0, count_buffer, 0, Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count, sizeof(VkDrawIndexedIndirectCommand));
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
			group.each([this, &cmd_buffer, &instance_id, state](const entt::entity &entity, const cmpt::DynamicMeshRenderer &mesh_renderer, const cmpt::Transform &transform, const cmpt::Tag &tag) {
				if (mesh_renderer.vertex_buffer && mesh_renderer.index_buffer)
				{
					Renderer::instance()->Render_Stats.dynamic_mesh_count.instance_count++;
					Renderer::instance()->Render_Stats.dynamic_mesh_count.triangle_count += static_cast<uint32_t>(mesh_renderer.indices.size()) / 3;

					VkDeviceSize offsets[1] = {0};
					vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &mesh_renderer.vertex_buffer.getBuffer(), offsets);
					vkCmdBindIndexBuffer(cmd_buffer, mesh_renderer.index_buffer, 0, VK_INDEX_TYPE_UINT32);

					m_vertex_block.dynamic   = 1;
					m_vertex_block.transform = transform.world_transform;
					m_vertex_block.entity_id = static_cast<uint32_t>(entity);
					vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(m_vertex_block), &m_vertex_block);

					vkCmdDrawIndexed(cmd_buffer, static_cast<uint32_t>(mesh_renderer.index_buffer.getSize() / sizeof(uint32_t)), 1, 0, 0, instance_id++);
				}
			});
		}
	}

	vkCmdEndRenderPass(cmd_buffer);
}
}        // namespace Ilum::pass