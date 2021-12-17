#include "DynamicGeometryPass.hpp"

#include "Renderer/Renderer.hpp"

#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "Threading/ThreadPool.hpp"

#include "Material/DisneyPBR.h"

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

	state.color_blend_attachment_states.resize(8);
	state.depth_stencil_state.stencil_test_enable = false;

	// Disable blending
	for (auto &color_blend_attachment_state : state.color_blend_attachment_states)
	{
		color_blend_attachment_state.blend_enable = false;
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

	state.addOutputAttachment("gbuffer - albedo", AttachmentState::Clear_Color);
	state.addOutputAttachment("gbuffer - normal", AttachmentState::Clear_Color);
	state.addOutputAttachment("gbuffer - position", AttachmentState::Clear_Color);
	state.addOutputAttachment("gbuffer - metallic_roughness_ao", AttachmentState::Clear_Color);
	state.addOutputAttachment("gbuffer - emissive", AttachmentState::Clear_Color);
	state.addOutputAttachment("debug - instance", AttachmentState::Clear_Color);

	VkClearColorValue clear_entity_id = {};
	clear_entity_id.uint32[0]         = static_cast<uint32_t>(entt::null);

	state.addOutputAttachment("debug - entity", clear_entity_id);

	state.addOutputAttachment("depth_stencil", AttachmentState::Load_Depth_Stencil);
}

void DynamicGeometryPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("Camera", Renderer::instance()->getBuffer(Renderer::BufferType::MainCamera));
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

	const auto group = Scene::instance()->getRegistry().group<>(entt::get<cmpt::MeshRenderer, cmpt::Transform, cmpt::Tag>);

	Renderer::instance()->Instance_Count = Renderer::instance()->Static_Instance_Count + group.size();

	group.each([&cmd_buffer](const entt::entity &entity, const cmpt::MeshRenderer &mesh_renderer, const cmpt::Transform &transform, const cmpt::Tag &tag) {
		if (mesh_renderer.vertex_buffer && mesh_renderer.index_buffer)
		{
			Renderer::instance()->Instance_Count++;

			VkDeviceSize offsets[1] = {0};
			vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &mesh_renderer.vertex_buffer.getBuffer(), offsets);
			vkCmdBindIndexBuffer(cmd_buffer, mesh_renderer.index_buffer, 0, VK_INDEX_TYPE_UINT32);

			struct VertexPushBlock
			{
				glm::mat4 transform;
				float     displacement_height;
				uint32_t  displacement_map;
				uint32_t  instance_id;
			}vertex_block;

			struct FragmentPushBlock
			{
				glm::vec4  base_color;
				glm::vec3  emissive_color;
				float metallic_factor;
				float roughness_factor;
				float emissive_intensity;

				uint32_t albedo_map;
				uint32_t normal_map;
				uint32_t metallic_map;
				uint32_t roughness_map;
				uint32_t emissive_map;
				uint32_t ao_map;
				uint32_t entity_id;
			};

		}
	});

	//auto &vertex_buffer = Renderer::instance()->getBuffer(Renderer::BufferType::Vertex);
	//auto &index_buffer  = Renderer::instance()->getBuffer(Renderer::BufferType::Index);

	//if (Renderer::instance()->Meshlet_Count > 0 && vertex_buffer.get().getBuffer() && index_buffer.get().getBuffer())
	//{
	//	VkDeviceSize offsets[1] = {0};
	//	vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &vertex_buffer.get().getBuffer(), offsets);
	//	vkCmdBindIndexBuffer(cmd_buffer, index_buffer.get().getBuffer(), 0, VK_INDEX_TYPE_UINT32);

	//	auto &draw_buffer  = Renderer::instance()->Render_Queue.Command_Buffer;
	//	auto &count_buffer = Renderer::instance()->Render_Queue.Count_Buffer;
	//	vkCmdDrawIndexedIndirectCount(cmd_buffer, draw_buffer, 0, count_buffer, 0, Renderer::instance()->Meshlet_Count, sizeof(VkDrawIndexedIndirectCommand));
	//}

	vkCmdEndRenderPass(cmd_buffer);
}
}        // namespace Ilum::pass