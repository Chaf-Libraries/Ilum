#include "DynamicGeometryPass.hpp"

#include "Renderer/Renderer.hpp"

#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "Threading/ThreadPool.hpp"

#include "File/FileSystem.hpp"

namespace Ilum::pass
{
DynamicGeometryPass::DynamicGeometryPass()
{
}

void DynamicGeometryPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/Deferred/DynamicGeometry.vert", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/Deferred/DynamicGeometry.frag", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::GLSL);

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
	state.declareAttachment("gbuffer - linear_depth", VK_FORMAT_R32_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("debug - entity", VK_FORMAT_R32_UINT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("depth_stencil", VK_FORMAT_D32_SFLOAT_S8_UINT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);

	state.addOutputAttachment("gbuffer - albedo", AttachmentState::Load_Color);
	state.addOutputAttachment("gbuffer - normal", AttachmentState::Load_Color);
	state.addOutputAttachment("gbuffer - position", AttachmentState::Load_Color);
	state.addOutputAttachment("gbuffer - metallic_roughness_ao", AttachmentState::Load_Color);
	state.addOutputAttachment("gbuffer - emissive", AttachmentState::Load_Color);
	state.addOutputAttachment("gbuffer - linear_depth", AttachmentState::Load_Color);
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

	const auto group = Scene::instance()->getRegistry().group<cmpt::DynamicMeshRenderer>(entt::get<cmpt::Transform, cmpt::Tag>);

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

	Renderer::instance()->Render_Stats.dynamic_mesh_count.instance_count = 0;
	Renderer::instance()->Render_Stats.dynamic_mesh_count.triangle_count = 0;

	uint32_t instance_id = Renderer::instance()->Render_Stats.static_mesh_count.instance_count;

	group.each([&cmd_buffer, &instance_id, state](const entt::entity &entity, const cmpt::DynamicMeshRenderer &mesh_renderer, const cmpt::Transform &transform, const cmpt::Tag &tag) {
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
				float     displacement;
				uint32_t  displacement_map;
				uint32_t  instance_id;
			} vertex_block;

			struct FragmentPushBlock
			{
				glm::vec4 base_color;
				glm::vec3 emissive_color;
				float     metallic;
				float     roughness;
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

			vertex_block.displacement = mesh_renderer.material.displacement;
			vertex_block.displacement_map    = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(mesh_renderer.material.textures[TextureType::Displacement]));

			fragment_block.base_color         = mesh_renderer.material.base_color;
			fragment_block.emissive_color     = mesh_renderer.material.emissive_color;
			fragment_block.metallic    = mesh_renderer.material.metallic;
			fragment_block.roughness   = mesh_renderer.material.roughness;
			fragment_block.emissive_intensity = mesh_renderer.material.emissive_intensity;
			fragment_block.albedo_map         = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(mesh_renderer.material.textures[TextureType::BaseColor]));
			fragment_block.normal_map         = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(mesh_renderer.material.textures[TextureType::Normal]));
			fragment_block.metallic_map       = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(mesh_renderer.material.textures[TextureType::Metallic]));
			fragment_block.roughness_map      = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(mesh_renderer.material.textures[TextureType::Roughness]));
			fragment_block.emissive_map       = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(mesh_renderer.material.textures[TextureType::Emissive]));
			fragment_block.ao_map             = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(mesh_renderer.material.textures[TextureType::AmbientOcclusion]));

			vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VertexPushBlock), &vertex_block);
			vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 80, sizeof(FragmentPushBlock), &fragment_block);

			vkCmdDrawIndexed(cmd_buffer, static_cast<uint32_t>(mesh_renderer.indices.size()), 1, 0, 0, 0);
		}
	});

	vkCmdEndRenderPass(cmd_buffer);
}
}        // namespace Ilum::pass