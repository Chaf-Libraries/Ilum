#include "Renderer.hpp"

#include <Core/Input.hpp>
#include <Core/Time.hpp>

#include <RHI/Command.hpp>
#include <RHI/DescriptorState.hpp>
#include <RHI/FrameBuffer.hpp>
#include <RHI/PipelineState.hpp>

#include <Scene/Component/Camera.hpp>
#include <Scene/Component/Transform.hpp>
#include <Scene/Entity.hpp>
#include <Scene/Scene.hpp>

#include <Asset/AssetManager.hpp>

#include <imgui.h>

#include <IconsFontAwesome4.h>

namespace Ilum
{
Renderer::Renderer(RHIDevice *device, Scene *scene) :
    p_device(device),
    p_scene(scene),
    m_rg(device, *this),
    m_rg_builder(device, m_rg, *this)
{
	CreateSampler();
	KullaContyApprox();
	BRDFPreIntegration();
	EquirectangularToCubemap(nullptr);
}

Renderer::~Renderer()
{
}

void Renderer::Tick()
{
	p_present = nullptr;

	m_rg.Execute();
}

void Renderer::OnImGui(ImGuiContext &context)
{
	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	// Renderer Inspector
	ImGui::Begin("Renderer");

	ImGui::Text("Render Target Size: (%ld, %ld)", m_extent.width, m_extent.height);
	ImGui::Text("Viewport Size: (%ld, %ld)", m_viewport.width, m_viewport.height);
	ImGui::Text("FPS: %f", Timer::GetInstance().FrameRate());

	if (ImGui::TreeNode("Skybox"))
	{
		const char *const faces[]         = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};
		static int32_t    current_face_id = 0;
		ImGui::Combo("Face", &current_face_id, faces, 6);
		if (ImGui::BeginDragDropTarget())
		{
			if (const auto *pay_load = ImGui::AcceptDragDropPayload("Texture"))
			{
				ASSERT(pay_load->DataSize == sizeof(uint32_t));
				auto *tex = p_scene->GetAssetManager().GetTexture(*static_cast<uint32_t *>(pay_load->Data));
				EquirectangularToCubemap(tex);
			}
			ImGui::EndDragDropTarget();
		}
		ImGui::Image(context.TextureID(m_skybox->GetView(TextureViewDesc{VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, static_cast<uint32_t>(current_face_id), 1})), ImVec2(300, 300));
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("LUT"))
	{
		if (ImGui::TreeNode("Kulla Conty Energy"))
		{
			ImGui::Image(context.TextureID(GetPrecompute(PrecomputeType::KullaContyEnergy).GetView(view_desc)), ImVec2(300, 300));
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Kulla Conty Energy Average"))
		{
			ImGui::Image(context.TextureID(GetPrecompute(PrecomputeType::KullaContyAverage).GetView(view_desc)), ImVec2(300, 300));
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("BRDF PreIntegration"))
		{
			ImGui::Image(context.TextureID(GetPrecompute(PrecomputeType::BRDFPreIntegration).GetView(view_desc)), ImVec2(300, 300));
			ImGui::TreePop();
		}
		ImGui::TreePop();
	}

	// Render Graph Editor
	bool recompile = m_rg_builder.OnImGui(context);

	if (ImGui::TreeNode("Render Passes"))
	{
		m_rg.OnImGui(context);
		ImGui::TreePop();
	}
	ImGui::End();

	// Scene View
	ImGui::Begin("Present");
	if (p_present && !recompile)
	{
		m_viewport = VkExtent2D{
		    static_cast<uint32_t>(ImGui::GetContentRegionAvail().x),
		    static_cast<uint32_t>(ImGui::GetContentRegionAvail().y)};

		auto camera_entity = Entity(*p_scene, p_scene->GetMainCamera());
		if (camera_entity.IsValid())
		{
			camera_entity.GetComponent<cmpt::Camera>().aspect = static_cast<float>(m_viewport.width) / static_cast<float>(m_viewport.height);
		}

		TextureViewDesc desc  = {};
		desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
		desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
		desc.base_mip_level   = 0;
		desc.base_array_layer = 0;
		desc.level_count      = p_present->GetMipLevels();
		desc.layer_count      = p_present->GetLayerCount();
		ImGui::Image(context.TextureID(p_present->GetView(desc)), ImGui::GetContentRegionAvail());

		// Camera Controling
		auto main_camera = Entity(*p_scene, p_scene->GetMainCamera());
		if (ImGui::IsWindowFocused() && ImGui::IsWindowHovered() && main_camera.IsValid())
		{
			if (Input::GetInstance().IsMouseButtonPressed(MouseCode::ButtonRight))
			{
				auto &camera    = main_camera.GetComponent<cmpt::Camera>();
				auto &transform = main_camera.GetComponent<cmpt::Transform>();

				glm::vec2 delta = glm::vec2(p_device->GetWindow()->m_pos_delta_x, p_device->GetWindow()->m_pos_delta_y);

				camera.frame_count = 0;

				float yaw   = std::atan2f(-camera.view[2][2], -camera.view[0][2]);
				float pitch = std::asinf(-glm::clamp(camera.view[1][2], -1.f, 1.f));

				if (delta.x != 0.f)
				{
					yaw += 0.001f * Timer::GetInstance().DeltaTime() * delta.x;
					transform.rotation.y = -glm::degrees(yaw) - 90.f;
				}
				if (delta.y != 0.f)
				{
					pitch -= 0.001f * Timer::GetInstance().DeltaTime() * static_cast<float>(delta.y);
					transform.rotation.x = glm::degrees(pitch);
				}

				glm::vec3 forward = glm::vec3(0.f);
				forward.x         = std::cosf(pitch) * std::cosf(yaw);
				forward.y         = std::sinf(pitch);
				forward.z         = std::cosf(pitch) * std::sinf(yaw);
				forward           = glm::normalize(forward);

				glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3{0.f, 1.f, 0.f}));
				glm::vec3 up    = glm::normalize(glm::cross(right, forward));

				glm::vec3 direction = glm::vec3(0.f);

				if (Input::GetInstance().IsKeyPressed(KeyCode::W))
				{
					direction += forward;
				}
				if (Input::GetInstance().IsKeyPressed(KeyCode::S))
				{
					direction -= forward;
				}
				if (Input::GetInstance().IsKeyPressed(KeyCode::D))
				{
					direction += right;
				}
				if (Input::GetInstance().IsKeyPressed(KeyCode::A))
				{
					direction -= right;
				}
				if (Input::GetInstance().IsKeyPressed(KeyCode::Q))
				{
					direction += up;
				}
				if (Input::GetInstance().IsKeyPressed(KeyCode::E))
				{
					direction -= up;
				}

				float t = glm::clamp(0.2f, 0.f, 1.f);

				m_translate_velocity = glm::mix(m_translate_velocity, direction, t * t * (3.f - 2.f * t));

				transform.translation += m_translate_velocity * Timer::GetInstance().DeltaTime() * 0.005f;

				glm::mat4 related_transform = transform.world_transform * glm::inverse(transform.local_transform);
				transform.local_transform   = glm::scale(glm::translate(glm::mat4(1.f), transform.translation) * glm::mat4_cast(glm::qua<float>(glm::radians(transform.rotation))), transform.scale);
				transform.world_transform   = related_transform * transform.local_transform;
			}
		}
	}
	ImGui::End();

	ImGui::ShowDemoWindow();

	// Scene UI
	if (p_scene)
	{
		p_scene->OnImGui(context);
	}
}

Sampler &Renderer::GetSampler(SamplerType type)
{
	return *m_samplers[static_cast<size_t>(type)];
}

Texture &Renderer::GetPrecompute(PrecomputeType type)
{
	return *m_precomputes[static_cast<size_t>(type)];
}

Texture &Renderer::GetSkybox()
{
	return *m_skybox;
}

const VkExtent2D &Renderer::GetExtent() const
{
	return m_extent;
}

const VkExtent2D &Renderer::GetViewport() const
{
	return m_viewport;
}

Scene *Renderer::GetScene()
{
	return p_scene;
}

void Renderer::SetScene(Scene *scene)
{
	p_scene = scene;
}

void Renderer::SetPresent(Texture *present)
{
	p_present = present;
}

void Renderer::CreateSampler()
{
	m_samplers[static_cast<size_t>(SamplerType::PointClamp)]       = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_NEAREST});
	m_samplers[static_cast<size_t>(SamplerType::PointWarp)]        = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_NEAREST});
	m_samplers[static_cast<size_t>(SamplerType::BilinearClamp)]    = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_NEAREST});
	m_samplers[static_cast<size_t>(SamplerType::BilinearWarp)]     = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_NEAREST});
	m_samplers[static_cast<size_t>(SamplerType::TrilinearClamp)]   = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_LINEAR});
	m_samplers[static_cast<size_t>(SamplerType::TrilinearWarp)]    = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_LINEAR});
	m_samplers[static_cast<size_t>(SamplerType::AnisptropicClamp)] = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_LINEAR});
	m_samplers[static_cast<size_t>(SamplerType::AnisptropicWarp)]  = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_LINEAR});
}

void Renderer::KullaContyApprox()
{
	// Declare render target
	TextureDesc tex_desc  = {};
	tex_desc.width        = 1024;
	tex_desc.height       = 1024;
	tex_desc.depth        = 1;
	tex_desc.mips         = 1;
	tex_desc.layers       = 1;
	tex_desc.sample_count = VK_SAMPLE_COUNT_1_BIT;
	tex_desc.format       = VK_FORMAT_R16_SFLOAT;
	tex_desc.usage        = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

	m_precomputes[static_cast<size_t>(PrecomputeType::KullaContyEnergy)]  = std::make_unique<Texture>(p_device, tex_desc);
	m_precomputes[static_cast<size_t>(PrecomputeType::KullaContyAverage)] = std::make_unique<Texture>(p_device, tex_desc);

	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	// Setup PSO
	ShaderDesc kulla_conty_energy_shader  = {};
	kulla_conty_energy_shader.filename    = "./Source/Shaders/KullaConty.hlsl";
	kulla_conty_energy_shader.entry_point = "main";
	kulla_conty_energy_shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	kulla_conty_energy_shader.type        = ShaderType::HLSL;
	kulla_conty_energy_shader.macros.push_back("Energy");

	ShaderDesc kulla_conty_average_shader  = {};
	kulla_conty_average_shader.filename    = "./Source/Shaders/KullaConty.hlsl";
	kulla_conty_average_shader.entry_point = "main";
	kulla_conty_average_shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	kulla_conty_average_shader.type        = ShaderType::HLSL;
	kulla_conty_average_shader.macros.push_back("Average");

	PipelineState Emu_pso;
	Emu_pso.LoadShader(kulla_conty_energy_shader);

	PipelineState Eavg_pso;
	Eavg_pso.LoadShader(kulla_conty_average_shader);

	// Record command buffer
	auto &cmd_buffer = p_device->RequestCommandBuffer();

	cmd_buffer.Begin();

	// Comput Emu
	{
		cmd_buffer.Transition(&GetPrecompute(PrecomputeType::KullaContyEnergy), TextureState{}, TextureState(VK_IMAGE_USAGE_STORAGE_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
		cmd_buffer.Bind(Emu_pso);
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, GetPrecompute(PrecomputeType::KullaContyEnergy).GetView(view_desc)));
		cmd_buffer.Dispatch(1024 / 32, 1024 / 32);
		cmd_buffer.Transition(&GetPrecompute(PrecomputeType::KullaContyEnergy), TextureState{VK_IMAGE_USAGE_STORAGE_BIT}, TextureState(VK_IMAGE_USAGE_SAMPLED_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
	}

	// Compute Eavg
	{
		cmd_buffer.Transition(&GetPrecompute(PrecomputeType::KullaContyAverage), TextureState{}, TextureState(VK_IMAGE_USAGE_STORAGE_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
		cmd_buffer.Bind(Eavg_pso);
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, GetPrecompute(PrecomputeType::KullaContyAverage).GetView(view_desc))
		        .Bind(0, 1, GetPrecompute(PrecomputeType::KullaContyEnergy).GetView(view_desc)));
		cmd_buffer.Dispatch(1024 / 32, 1024 / 32);
		cmd_buffer.Transition(&GetPrecompute(PrecomputeType::KullaContyAverage), TextureState{VK_IMAGE_USAGE_STORAGE_BIT}, TextureState(VK_IMAGE_USAGE_SAMPLED_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
	}

	cmd_buffer.End();

	// Submit
	p_device->SubmitIdle(cmd_buffer);
}

void Renderer::BRDFPreIntegration()
{
	// Declare render target
	TextureDesc tex_desc  = {};
	tex_desc.width        = 512;
	tex_desc.height       = 512;
	tex_desc.depth        = 1;
	tex_desc.mips         = 1;
	tex_desc.layers       = 1;
	tex_desc.sample_count = VK_SAMPLE_COUNT_1_BIT;
	tex_desc.format       = VK_FORMAT_R16G16_SFLOAT;
	tex_desc.usage        = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

	m_precomputes[static_cast<size_t>(PrecomputeType::BRDFPreIntegration)] = std::make_unique<Texture>(p_device, tex_desc);

	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	// Setup PSO
	ShaderDesc brdf_preintegration_shader  = {};
	brdf_preintegration_shader.filename    = "./Source/Shaders/BRDFPreIntegration.hlsl";
	brdf_preintegration_shader.entry_point = "main";
	brdf_preintegration_shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	brdf_preintegration_shader.type        = ShaderType::HLSL;

	PipelineState pso;
	pso.LoadShader(brdf_preintegration_shader);

	// Record command buffer
	auto &cmd_buffer = p_device->RequestCommandBuffer();

	cmd_buffer.Begin();

	// Comput BRDF Preintegration
	{
		cmd_buffer.Transition(&GetPrecompute(PrecomputeType::BRDFPreIntegration), TextureState{}, TextureState(VK_IMAGE_USAGE_STORAGE_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
		cmd_buffer.Bind(pso);
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, GetPrecompute(PrecomputeType::BRDFPreIntegration).GetView(view_desc)));
		cmd_buffer.Dispatch(512 / 32, 512 / 32);
		cmd_buffer.Transition(&GetPrecompute(PrecomputeType::BRDFPreIntegration), TextureState{VK_IMAGE_USAGE_STORAGE_BIT}, TextureState(VK_IMAGE_USAGE_SAMPLED_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
	}

	cmd_buffer.End();

	// Submit
	p_device->SubmitIdle(cmd_buffer);
}

void Renderer::EquirectangularToCubemap(Texture *texture)
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

	if (!m_skybox)
	{
		m_skybox = std::make_unique<Texture>(p_device, tex_desc);

		auto &cmd_buffer = p_device->RequestCommandBuffer();
		cmd_buffer.Begin();
		cmd_buffer.Transition(m_skybox.get(), TextureState{}, TextureState(VK_IMAGE_USAGE_SAMPLED_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6});
		cmd_buffer.End();
		p_device->SubmitIdle(cmd_buffer);

		m_skybox->SetName("Skybox");
	}

	if (!texture)
	{
		return;
	}

	// Setup PSO
	ShaderDesc vertex_shader  = {};
	vertex_shader.filename    = "./Source/Shaders/Precompute/EquirectangularToCubemap.hlsl";
	vertex_shader.entry_point = "VSmain";
	vertex_shader.stage       = VK_SHADER_STAGE_VERTEX_BIT;
	vertex_shader.type        = ShaderType::HLSL;

	ShaderDesc fragment_shader  = {};
	fragment_shader.filename    = "./Source/Shaders/Precompute/EquirectangularToCubemap.hlsl";
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
	auto &cmd_buffer = p_device->RequestCommandBuffer();
	cmd_buffer.Begin();
	cmd_buffer.Transition(m_skybox.get(), TextureState(VK_IMAGE_USAGE_SAMPLED_BIT), TextureState(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6});
	for (uint32_t i = 0; i < 6; i++)
	{
		FrameBuffer frame_buffer = {};
		frame_buffer.Bind(
		    m_skybox.get(),
		    TextureViewDesc{
		        VK_IMAGE_VIEW_TYPE_2D,
		        VK_IMAGE_ASPECT_COLOR_BIT,
		        0, 1, i, 1},
		    ColorAttachmentInfo{});
		cmd_buffer.BeginRenderPass(frame_buffer);

		cmd_buffer.Bind(pso);
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, texture->GetView(view_desc))
		        .Bind(0, 1, *m_samplers[static_cast<uint32_t>(SamplerType::TrilinearClamp)]));

		auto push_data = glm::inverse(projection_matrix * views_matrix[i]);

		cmd_buffer.SetViewport(1024, 1024);
		cmd_buffer.SetScissor(1024, 1024);
		cmd_buffer.PushConstants(VK_SHADER_STAGE_VERTEX_BIT, &push_data, sizeof(glm::mat4), 0);
		cmd_buffer.Draw(3, 1, 0, 0);

		cmd_buffer.EndRenderPass();
	}
	cmd_buffer.Transition(m_skybox.get(), TextureState{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT}, TextureState(VK_IMAGE_USAGE_SAMPLED_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6});
	cmd_buffer.End();

	// Submit
	p_device->SubmitIdle(cmd_buffer);
}

}        // namespace Ilum