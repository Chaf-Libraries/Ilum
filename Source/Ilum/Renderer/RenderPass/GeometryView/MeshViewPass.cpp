#include "MeshViewPass.hpp"

#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"

#include "Renderer/Renderer.hpp"

#include "Editor/Editor.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "File/FileSystem.hpp"

#include <imgui.h>

namespace Ilum::pass
{
void MeshViewPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/GeometryView/MeshView.hlsl", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::HLSL, "VSmain");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/GeometryView/MeshView.hlsl", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::HLSL,"PSmain");

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
	state.descriptor_bindings.bind(0, 1, "TextureArray", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Wrap), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	state.declareAttachment("GeometryView", VK_FORMAT_R8G8B8A8_UNORM, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("GeometryDepthStencil", VK_FORMAT_D32_SFLOAT_S8_UINT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);

	state.addOutputAttachment("GeometryView", AttachmentState::Load_Color);
	state.addOutputAttachment("GeometryDepthStencil", AttachmentState::Load_Depth_Stencil);
}

void MeshViewPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
	resolve.resolve("TextureArray", Renderer::instance()->getResourceCache().getImageReferences());
}

void MeshViewPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	auto entity = Editor::instance()->getSelect();

	if (entity && entity.hasComponent<cmpt::DynamicMeshRenderer>() && entity.getComponent<cmpt::DynamicMeshRenderer>().vertex_buffer)
	{
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

		const auto &mesh = entity.getComponent<cmpt::DynamicMeshRenderer>();

		VkDeviceSize offsets[1] = {0};
		vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &mesh.vertex_buffer.getBuffer(), offsets);
		vkCmdBindIndexBuffer(cmd_buffer, mesh.index_buffer.getBuffer(), 0, VK_INDEX_TYPE_UINT32);

		struct
		{
			glm::mat4 transform;
			uint32_t  texture_id;
			uint32_t  parameterization;
		} push_block;

		push_block.transform        = entity.getComponent<cmpt::Transform>().world_transform;
		push_block.texture_id       = m_texture_id;
		push_block.parameterization = m_parameterization;

		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push_block), &push_block);
		vkCmdDrawIndexed(cmd_buffer, static_cast<uint32_t>(mesh.index_buffer.getSize()) / sizeof(uint32_t), 1, 0, 0, 0);

		vkCmdEndRenderPass(cmd_buffer);
	}
}

void MeshViewPass::onImGui()
{
	ImGui::Checkbox("Show Parameterization", reinterpret_cast<bool *>(&m_parameterization));

	ImGui::PushID("Mesh Texture");
	if (ImGui::ImageButton(Renderer::instance()->getResourceCache().hasImage(FileSystem::getRelativePath(m_texture)) ?
                               ImGuiContext::textureID(Renderer::instance()->getResourceCache().loadImage(FileSystem::getRelativePath(m_texture)), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)) :
                               ImGuiContext::textureID(Renderer::instance()->getDefaultTexture(), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
	                       ImVec2{100.f, 100.f}))
	{
		m_texture    = "";
		m_texture_id = std::numeric_limits<uint32_t>::max();
	}
	ImGui::PopID();

	if (ImGui::BeginDragDropTarget())
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Texture2D"))
		{
			ASSERT(pay_load->DataSize == sizeof(std::string));
			if (m_texture != *static_cast<std::string *>(pay_load->Data))
			{
				m_texture    = *static_cast<std::string *>(pay_load->Data);
				m_texture_id = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(m_texture));
			}
		}
		ImGui::EndDragDropTarget();
	}
}
}        // namespace Ilum::pass