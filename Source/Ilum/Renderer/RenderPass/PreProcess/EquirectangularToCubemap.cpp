#include "EquirectangularToCubemap.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Vulkan/VK_Debugger.h"

#include "ImGui/ImGuiContext.hpp"

#include "File/FileSystem.hpp"

#include "Device/LogicalDevice.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>

namespace Ilum::pass
{
EquirectangularToCubemap::~EquirectangularToCubemap()
{
	for (auto &framebuffer : m_framebuffers)
	{
		vkDestroyFramebuffer(GraphicsContext::instance()->getLogicalDevice(), framebuffer, nullptr);
	}
}

void EquirectangularToCubemap::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/EquirectangularToCubemap.vert", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/EquirectangularToCubemap.frag", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::GLSL);

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR};

	state.color_blend_attachment_states.resize(1);
	state.depth_stencil_state.stencil_test_enable = false;

	// Disable blending
	for (auto &color_blend_attachment_state : state.color_blend_attachment_states)
	{
		color_blend_attachment_state.blend_enable = false;
	}

	state.rasterization_state.polygon_mode = VK_POLYGON_MODE_FILL;

	state.descriptor_bindings.bind(0, 0, "TextureArray", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Wrap), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	state.declareAttachment("SkyBox", VK_FORMAT_R16G16B16A16_SFLOAT, 1024, 1024, false, 6);

	state.addOutputAttachment("SkyBox", AttachmentState::Clear_Color);
}

void EquirectangularToCubemap::resolveResources(ResolveState &resolve)
{
	resolve.resolve("TextureArray", Renderer::instance()->getResourceCache().getImageReferences());
}

void EquirectangularToCubemap::render(RenderPassState &state)
{
	if (!m_update)
	{
		return;
	}

	auto &cmd_buffer = state.command_buffer;

	auto &attachment = Renderer::instance()->getRenderGraph()->getAttachment("SkyBox");

	if (m_framebuffers.empty())
	{
		VK_Debugger::setName(attachment, "cubemap");

		m_framebuffers.resize(6);

		for (uint32_t layer = 0; layer < 6; layer++)
		{
			VkFramebufferCreateInfo frame_buffer_create_info = {};
			frame_buffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			frame_buffer_create_info.renderPass              = state.pass.render_pass;
			frame_buffer_create_info.attachmentCount         = 1;
			frame_buffer_create_info.pAttachments            = &attachment.getView(layer);
			frame_buffer_create_info.width                   = 1024;
			frame_buffer_create_info.height                  = 1024;
			frame_buffer_create_info.layers                  = 1;

			vkCreateFramebuffer(GraphicsContext::instance()->getLogicalDevice(), &frame_buffer_create_info, nullptr, &m_framebuffers[layer]);
		}
	}

	VkRenderPassBeginInfo begin_info = {};
	begin_info.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	begin_info.renderPass            = state.pass.render_pass;
	begin_info.renderArea            = state.pass.render_area;
	begin_info.framebuffer           = state.pass.frame_buffer;
	begin_info.clearValueCount       = static_cast<uint32_t>(state.pass.clear_values.size());
	begin_info.pClearValues          = state.pass.clear_values.data();

	VkViewport viewport = {0, 0, 1024, 1024, 0, 1};
	VkRect2D   scissor  = {0, 0, 1024, 1024};

	glm::mat4 projection_matrix = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
	glm::mat4 views_matrix[] =
	    {
	        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
	        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
	        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
	        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
	        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
	        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))};

	uint32_t texID = Renderer::instance()->getResourceCache().imageID(m_filename);

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	if (!Renderer::instance()->getResourceCache().hasImage(m_filename))
	{
		for (uint32_t i = 0; i < 6; i++)
		{
			begin_info.framebuffer = m_framebuffers[i];
			vkCmdBeginRenderPass(cmd_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdEndRenderPass(cmd_buffer);
		}

		m_update = false;
		return;
	}

	for (uint32_t i = 0; i < 6; i++)
	{
		begin_info.framebuffer = m_framebuffers[i];
		vkCmdBeginRenderPass(cmd_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

		glm::mat4 view_projection = projection_matrix * views_matrix[i];

		vkCmdSetViewport(cmd_buffer, 0, 1, &viewport);
		vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);

		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &view_projection);
		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(uint32_t), &texID);

		vkCmdDraw(cmd_buffer, 3, 1, 0, 0);

		vkCmdEndRenderPass(cmd_buffer);
	}

	m_update = false;
}

void EquirectangularToCubemap::onImGui()
{
	ImGui::PushID("Environment Light");
	if (ImGui::ImageButton(Renderer::instance()->getResourceCache().hasImage(FileSystem::getRelativePath(m_filename)) ?
                               ImGuiContext::textureID(Renderer::instance()->getResourceCache().loadImage(FileSystem::getRelativePath(m_filename)), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)) :
                               ImGuiContext::textureID(Renderer::instance()->getDefaultTexture(), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
	                       ImVec2{100.f, 100.f}))
	{
		m_filename = "";
		m_update   = true;
	}
	ImGui::PopID();

	if (ImGui::BeginDragDropTarget())
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Texture2D"))
		{
			ASSERT(pay_load->DataSize == sizeof(std::string));
			if (m_filename != *static_cast<std::string *>(pay_load->Data))
			{
				m_filename = *static_cast<std::string *>(pay_load->Data);
				m_update   = true;
			}
		}
		ImGui::EndDragDropTarget();
	}
}
}        // namespace Ilum::pass