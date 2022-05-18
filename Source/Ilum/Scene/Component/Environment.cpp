#include "Environment.hpp"

#include "Scene/Scene.hpp"

#include <RHI/Command.hpp>
#include <RHI/DescriptorState.hpp>
#include <RHI/Device.hpp>
#include <RHI/FrameBuffer.hpp>
#include <RHI/PipelineState.hpp>
#include <RHI/Sampler.hpp>
#include <RHI/Texture.hpp>

namespace Ilum::cmpt
{
void Environment::Tick(Scene &scene, entt::entity entity, RHIDevice *device)
{
	m_manager = &scene.GetAssetManager();

	if (m_update)
	{
		if (m_type == EnvironmentType::Skybox)
		{
			// Declare render target
			TextureDesc tex_desc  = {};
			tex_desc.width        = 1024;
			tex_desc.height       = 1024;
			tex_desc.depth        = 1;
			tex_desc.mips         = 1;
			tex_desc.layers       = 6;
			tex_desc.sample_count = VK_SAMPLE_COUNT_1_BIT;
			tex_desc.format       = VK_FORMAT_R16G16B16A16_SFLOAT;
			tex_desc.usage        = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

			TextureViewDesc view_desc  = {};
			view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
			view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
			view_desc.base_array_layer = 0;
			view_desc.base_mip_level   = 0;
			view_desc.layer_count      = 1;
			view_desc.level_count      = 1;

			if (!m_cubemap)
			{
				m_cubemap = std::make_unique<Texture>(device, tex_desc);

				auto &cmd_buffer = device->RequestCommandBuffer();
				cmd_buffer.Begin();
				cmd_buffer.Transition(m_cubemap.get(), TextureState{}, TextureState(VK_IMAGE_USAGE_SAMPLED_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6});
				cmd_buffer.End();
				device->SubmitIdle(cmd_buffer);
				m_cubemap->SetName("Skybox");
			}

			if (!scene.GetAssetManager().IsValid(m_skybox))
			{
				m_update = false;
				return;
			}

			// Sampler
			auto sampler = std::make_unique<Sampler>(device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_LINEAR});

			// Setup PSO
			ShaderDesc vertex_shader  = {};
			vertex_shader.filename    = "./Source/Shaders/EquirectangularToCubemap.hlsl";
			vertex_shader.entry_point = "VSmain";
			vertex_shader.stage       = VK_SHADER_STAGE_VERTEX_BIT;
			vertex_shader.type        = ShaderType::HLSL;

			ShaderDesc fragment_shader  = {};
			fragment_shader.filename    = "./Source/Shaders/EquirectangularToCubemap.hlsl";
			fragment_shader.entry_point = "PSmain";
			fragment_shader.stage       = VK_SHADER_STAGE_FRAGMENT_BIT;
			fragment_shader.type        = ShaderType::HLSL;

			ColorBlendState blend_state = {};
			blend_state.attachment_states.resize(1);

			DynamicState dynamic_state   = {};
			dynamic_state.dynamic_states = {
			    VK_DYNAMIC_STATE_VIEWPORT,
			    VK_DYNAMIC_STATE_SCISSOR};

			PipelineState pso;
			pso.LoadShader(vertex_shader);
			pso.LoadShader(fragment_shader);
			pso.SetDynamicState(dynamic_state);
			pso.SetColorBlendState(blend_state);

			glm::mat4 projection_matrix = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
			glm::mat4 views_matrix[] =
			    {
			        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
			        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
			        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
			        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
			        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
			        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))};

			// Record command buffer
			auto &cmd_buffer = device->RequestCommandBuffer();
			cmd_buffer.Begin();
			cmd_buffer.Transition(m_cubemap.get(), TextureState(VK_IMAGE_USAGE_SAMPLED_BIT), TextureState(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6});
			for (uint32_t i = 0; i < 6; i++)
			{
				FrameBuffer frame_buffer = {};
				frame_buffer.Bind(
				    m_cubemap.get(),
				    TextureViewDesc{
				        VK_IMAGE_VIEW_TYPE_2D,
				        VK_IMAGE_ASPECT_COLOR_BIT,
				        0, 1, i, 1},
				    ColorAttachmentInfo{});
				cmd_buffer.BeginRenderPass(frame_buffer);

				cmd_buffer.Bind(pso);
				cmd_buffer.Bind(
				    cmd_buffer.GetDescriptorState()
				        .Bind(0, 0, m_skybox->GetView(view_desc))
				        .Bind(0, 1, *sampler));

				auto push_data = glm::inverse(projection_matrix * views_matrix[i]);

				cmd_buffer.SetViewport(1024, 1024);
				cmd_buffer.SetScissor(1024, 1024);
				cmd_buffer.PushConstants(VK_SHADER_STAGE_VERTEX_BIT, &push_data, sizeof(glm::mat4), 0);
				cmd_buffer.Draw(3, 1, 0, 0);

				cmd_buffer.EndRenderPass();
			}
			cmd_buffer.Transition(m_cubemap.get(), TextureState{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT}, TextureState(VK_IMAGE_USAGE_SAMPLED_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6});
			cmd_buffer.End();

			// Submit
			device->SubmitIdle(cmd_buffer);
		}

		m_update = false;
	}
}

bool Environment::OnImGui(ImGuiContext &context)
{
	const char *const types[] = {"Skybox", "Procedural"};
	ImGui::Combo("Type", reinterpret_cast<int32_t *>(&m_type), types, 2);

	if (m_type == EnvironmentType::Skybox)
	{
		const char *const faces[]         = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};
		static int32_t    current_face_id = 0;
		ImGui::Combo("Face", &current_face_id, faces, 6);
		if (ImGui::BeginDragDropTarget())
		{
			if (const auto *pay_load = ImGui::AcceptDragDropPayload("Texture"))
			{
				ASSERT(pay_load->DataSize == sizeof(uint32_t));
				m_skybox = m_manager->GetTexture(*static_cast<uint32_t *>(pay_load->Data));
				Update();
			}
			ImGui::EndDragDropTarget();
		}
		if (m_cubemap)
		{
			ImGui::Image(context.TextureID(m_cubemap->GetView(TextureViewDesc{VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, static_cast<uint32_t>(current_face_id), 1})), ImVec2(300, 300));
		}
	}
	
	return m_update;
}

Texture *Environment::GetCubemap()
{
	return m_cubemap.get();
}

}        // namespace Ilum::cmpt
