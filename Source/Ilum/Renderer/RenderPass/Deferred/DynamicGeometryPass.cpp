#include "DynamicGeometryPass.hpp"

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
DynamicGeometryPass::DynamicGeometryPass()
{
}

void DynamicGeometryPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/dynamic_geometry.vert", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/dynamic_geometry.frag", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::GLSL);

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

	state.color_blend_attachment_states.resize(7);
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

	state.descriptor_bindings.bind(0, 0, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	state.descriptor_bindings.bind(0, 1, "textureArray", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Wrap), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	state.declareAttachment("gbuffer - albedo", VK_FORMAT_R8G8B8A8_UNORM, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("gbuffer - normal", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("gbuffer - position", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("gbuffer - metallic_roughness_ao", VK_FORMAT_R8G8B8A8_UNORM, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("gbuffer - emissive", VK_FORMAT_R8G8B8A8_UNORM, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("debug - instance", VK_FORMAT_R8G8B8A8_UNORM, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("debug - entity", VK_FORMAT_R32_UINT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("depth_stencil", VK_FORMAT_D32_SFLOAT_S8_UINT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);

	state.addOutputAttachment("gbuffer - albedo", AttachmentState::Load_Color);
	state.addOutputAttachment("gbuffer - normal", AttachmentState::Load_Color);
	state.addOutputAttachment("gbuffer - position", AttachmentState::Load_Color);
	state.addOutputAttachment("gbuffer - metallic_roughness_ao", AttachmentState::Load_Color);
	state.addOutputAttachment("gbuffer - emissive", AttachmentState::Load_Color);
	state.addOutputAttachment("debug - instance", AttachmentState::Load_Color);
	state.addOutputAttachment("debug - entity", AttachmentState::Load_Color);
	state.addOutputAttachment("depth_stencil", AttachmentState::Load_Depth_Stencil);
}

void DynamicGeometryPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
	resolve.resolve("textureArray", Renderer::instance()->getResourceCache().getImageReferences());
}

void DynamicGeometryPass::render(RenderPassState &state)
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

	const auto group = Scene::instance()->getRegistry().group<cmpt::MeshRenderer>(entt::get<cmpt::Transform, cmpt::Tag>);

	Renderer::instance()->Render_Stats.dynamic_mesh_count.instance_count = 0;
	Renderer::instance()->Render_Stats.dynamic_mesh_count.triangle_count = 0;

	uint32_t instance_id = Renderer::instance()->Render_Stats.static_mesh_count.instance_count;

	group.each([&cmd_buffer, &instance_id, state](const entt::entity &entity, const cmpt::MeshRenderer &mesh_renderer, const cmpt::Transform &transform, const cmpt::Tag &tag) {
		if (mesh_renderer.vertex_buffer && mesh_renderer.index_buffer)
		{
			Renderer::instance()->Render_Stats.dynamic_mesh_count.instance_count++;
			Renderer::instance()->Render_Stats.dynamic_mesh_count.triangle_count += static_cast<uint32_t>(mesh_renderer.indices.size()) / 3;

			VkDeviceSize offsets[1] = {0};
			vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &mesh_renderer.vertex_buffer.getBuffer(), offsets);
			vkCmdBindIndexBuffer(cmd_buffer, mesh_renderer.index_buffer, 0, VK_INDEX_TYPE_UINT32);

			struct VertexPushBlock
			{
				glm::mat4 transform;
				float displacement_height;
				uint32_t  displacement_map;
				uint32_t instance_id;
			} vertex_block;

			struct FragmentPushBlock
			{
				glm::vec4 base_color;
				glm::vec3 emissive_color;
				float     metallic_factor;
				float     roughness_factor;
				float     emissive_intensity;

				uint32_t albedo_map;
				uint32_t normal_map;
				uint32_t metallic_map;
				uint32_t roughness_map;
				uint32_t emissive_map;
				uint32_t ao_map;
				uint32_t entity_id;
			} fragment_block;

			vertex_block.transform   = transform.world_transform;
			vertex_block.instance_id = instance_id++;

			fragment_block.entity_id = static_cast<uint32_t>(entity);

			if (mesh_renderer.material && mesh_renderer.material->type() == typeid(material::PBRMaterial))
			{
				material::PBRMaterial *pbr = static_cast<material::PBRMaterial *>(mesh_renderer.material.get());

				vertex_block.displacement_height = pbr->displacement_height;
				vertex_block.displacement_map    = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(pbr->displacement_map));

				fragment_block.base_color         = pbr->base_color;
				fragment_block.emissive_color     = pbr->emissive_color;
				fragment_block.metallic_factor    = pbr->metallic_factor;
				fragment_block.roughness_factor   = pbr->roughness_factor;
				fragment_block.emissive_intensity = pbr->emissive_intensity;
				fragment_block.albedo_map         = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(pbr->albedo_map));
				fragment_block.normal_map         = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(pbr->normal_map));
				fragment_block.metallic_map       = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(pbr->metallic_map));
				fragment_block.roughness_map      = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(pbr->roughness_map));
				fragment_block.emissive_map       = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(pbr->emissive_map));
				fragment_block.ao_map             = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(pbr->ao_map));
			}
			else
			{
				vertex_block.displacement_height = 0.f;
				vertex_block.displacement_map    = std::numeric_limits<uint32_t>::max();

				fragment_block.base_color         = glm::vec4(1.f);
				fragment_block.emissive_color     = glm::vec4(0.f);
				fragment_block.metallic_factor    = 1.f;
				fragment_block.roughness_factor   = 1.f;
				fragment_block.emissive_intensity = 0.f;
				fragment_block.albedo_map         = std::numeric_limits<uint32_t>::max();
				fragment_block.normal_map         = std::numeric_limits<uint32_t>::max();
				fragment_block.metallic_map       = std::numeric_limits<uint32_t>::max();
				fragment_block.roughness_map      = std::numeric_limits<uint32_t>::max();
				fragment_block.emissive_map       = std::numeric_limits<uint32_t>::max();
				fragment_block.ao_map             = std::numeric_limits<uint32_t>::max();
			}

			vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VertexPushBlock), &vertex_block);
			vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 80, sizeof(FragmentPushBlock), &fragment_block);

			vkCmdDrawIndexed(cmd_buffer, static_cast<uint32_t>(mesh_renderer.indices.size()), 1, 0, 0, 0);
		}
	});

	vkCmdEndRenderPass(cmd_buffer);
}
}        // namespace Ilum::pass