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

#include <ImGuizmo.h>

#include <glm/gtc/type_ptr.hpp>

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
	GenerateLUT();

	p_device->WaitIdle();
}

Renderer::~Renderer()
{
}

void Renderer::Tick()
{
	p_present = nullptr;

	m_rg.Execute();

	//DrawPrimitive();
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
		if (ImGui::TreeNode("GGX LUT"))
		{
			ImGui::Image(context.TextureID(m_ggx_lut->GetView(view_desc)), ImVec2(300, 300));
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Charlie LUT"))
		{
			ImGui::Image(context.TextureID(m_charlie_lut->GetView(view_desc)), ImVec2(300, 300));
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

	ImGuizmo::SetDrawlist();
	auto offset              = ImGui::GetCursorPos();
	auto scene_view_size     = ImVec2(ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x, ImGui::GetWindowContentRegionMax().y - ImGui::GetWindowContentRegionMin().y);
	auto scene_view_position = ImVec2(ImGui::GetWindowPos().x + offset.x, ImGui::GetWindowPos().y + offset.y);

	scene_view_size.x -= static_cast<uint32_t>(scene_view_size.x) % 2 != 0 ? 1.0f : 0.0f;
	scene_view_size.y -= static_cast<uint32_t>(scene_view_size.y) % 2 != 0 ? 1.0f : 0.0f;

	ImGuizmo::SetRect(scene_view_position.x, scene_view_position.y, scene_view_size.x, scene_view_size.y);

	if (p_present && !recompile)
	{
		m_viewport = VkExtent2D{
		    static_cast<uint32_t>(ImGui::GetContentRegionAvail().x),
		    static_cast<uint32_t>(ImGui::GetContentRegionAvail().y)};

		auto camera_entity = Entity(*p_scene, p_scene->GetMainCamera());
		if (camera_entity.IsValid() && camera_entity.GetComponent<cmpt::Camera>().GetAspect() != static_cast<float>(m_viewport.width) / static_cast<float>(m_viewport.height))
		{
			camera_entity.GetComponent<cmpt::Camera>().SetAspect(static_cast<float>(m_viewport.width) / static_cast<float>(m_viewport.height));
		}

		// Display scene view
		{
			TextureViewDesc desc  = {};
			desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
			desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
			desc.base_mip_level   = 0;
			desc.base_array_layer = 0;
			desc.level_count      = p_present->GetMipLevels();
			desc.layer_count      = p_present->GetLayerCount();
			ImGui::Image(context.TextureID(p_present->GetView(desc)), ImGui::GetContentRegionAvail());
		}

		auto main_camera = Entity(*p_scene, p_scene->GetMainCamera());

		if (main_camera.IsValid() && main_camera.HasComponent<cmpt::Camera>())
		{
			auto &camera = main_camera.GetComponent<cmpt::Camera>();
			// ImGuizmo manipulate
			{
				auto selected = Entity(*p_scene, p_scene->GetSelected());
				if (p_scene && p_scene->GetRegistry().valid(p_scene->GetMainCamera()))
				{
					auto      e          = Entity(*p_scene, p_scene->GetMainCamera());
					glm::mat4 view       = camera.GetView();
					glm::mat4 projection = camera.GetProjection();

					if (selected.IsValid())
					{
						auto &transform = selected.GetComponent<cmpt::Transform>();

						glm::mat4 world_transform = transform.GetWorldTransform();

						bool is_on_guizmo = ImGuizmo::Manipulate(
						    glm::value_ptr(view),
						    glm::value_ptr(projection),
						    ImGuizmo::OPERATION::UNIVERSAL,
						    ImGuizmo::WORLD, glm::value_ptr(world_transform), NULL, NULL, NULL, NULL);

						if (is_on_guizmo)
						{
							glm::vec3 translation = {}, rotation = {}, scale = {};
							glm::mat4 local_transform = transform.GetLocalTransform() * glm::inverse(transform.GetWorldTransform()) * world_transform;
							ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(local_transform),
							                                      glm::value_ptr(translation),
							                                      glm::value_ptr(rotation),
							                                      glm::value_ptr(scale));
							transform.SetTranslation(translation);
							transform.SetRotation(rotation);
							transform.SetScale(scale);
						}
					}
				}
			}

			// Camera Controling
			if (ImGui::IsWindowFocused() && ImGui::IsWindowHovered())
			{
				if (Input::GetInstance().IsMouseButtonPressed(MouseCode::ButtonRight))
				{
					auto &transform = main_camera.GetComponent<cmpt::Transform>();

					glm::vec2 delta = glm::vec2(p_device->GetWindow()->m_pos_delta_x, p_device->GetWindow()->m_pos_delta_y);

					float yaw   = std::atan2f(-camera.GetView()[2][2], -camera.GetView()[0][2]);
					float pitch = std::asinf(-glm::clamp(camera.GetView()[1][2], -1.f, 1.f));

					glm::vec3 rotation    = transform.GetRotation();
					glm::vec3 translation = transform.GetTranslation();

					if (delta.x != 0.f)
					{
						yaw += 0.001f * Timer::GetInstance().DeltaTime() * delta.x;
						rotation.y = -glm::degrees(yaw) - 90.f;
					}
					if (delta.y != 0.f)
					{
						pitch -= 0.001f * Timer::GetInstance().DeltaTime() * static_cast<float>(delta.y);
						rotation.x = glm::degrees(pitch);
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

					constexpr float t = glm::clamp(0.2f, 0.f, 1.f);

					m_translate_velocity = glm::mix(m_translate_velocity, direction, t * t * (3.f - 2.f * t));

					translation += m_translate_velocity * Timer::GetInstance().DeltaTime() * 0.005f;

					transform.SetTranslation(translation);
					transform.SetRotation(rotation);

					transform.Update();
					camera.Update();
				}
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

void Renderer::SetDepthStencil(Texture *depth_stencil)
{
	p_depth_stencil = depth_stencil;
}

void Renderer::CreateSampler()
{
	m_samplers[static_cast<size_t>(SamplerType::PointClamp)]       = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_NEAREST});
	m_samplers[static_cast<size_t>(SamplerType::PointWarp)]        = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_NEAREST});
	m_samplers[static_cast<size_t>(SamplerType::BilinearClamp)]    = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_NEAREST});
	m_samplers[static_cast<size_t>(SamplerType::BilinearWarp)]     = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_NEAREST});
	m_samplers[static_cast<size_t>(SamplerType::TrilinearClamp)]   = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_LINEAR});
	m_samplers[static_cast<size_t>(SamplerType::TrilinearWarp)]    = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_LINEAR});
	m_samplers[static_cast<size_t>(SamplerType::AnisptropicClamp)] = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_LINEAR, true});
	m_samplers[static_cast<size_t>(SamplerType::AnisptropicWarp)]  = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_LINEAR, true});
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

void Renderer::GenerateLUT()
{
	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	m_ggx_lut = std::make_unique<Texture>(p_device, TextureDesc{
	                                                    /*width*/ 512,
	                                                    /*height*/ 512,
	                                                    /*depth*/ 1,
	                                                    /*mips*/ 1,
	                                                    /*layers*/ 1,
	                                                    /*sample_count*/ VK_SAMPLE_COUNT_1_BIT,
	                                                    /*format*/ VK_FORMAT_R16G16_SFLOAT,
	                                                    /*usage*/ VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT});

	m_charlie_lut = std::make_unique<Texture>(p_device, TextureDesc{
	                                                        /*width*/ 512,
	                                                        /*height*/ 512,
	                                                        /*depth*/ 1,
	                                                        /*mips*/ 1,
	                                                        /*layers*/ 1,
	                                                        /*sample_count*/ VK_SAMPLE_COUNT_1_BIT,
	                                                        /*format*/ VK_FORMAT_R16_SFLOAT,
	                                                        /*usage*/ VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT});

	ShaderDesc brdf_preintegration_shader  = {};
	brdf_preintegration_shader.filename    = "./Source/Shaders/GenerateLUT.hlsl";
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
		cmd_buffer.Transition({},
		                      {TextureTransition{
		                           m_ggx_lut.get(),
		                           TextureState{},
		                           TextureState{VK_IMAGE_USAGE_STORAGE_BIT},
		                           VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
		                       TextureTransition{
		                           m_charlie_lut.get(),
		                           TextureState{},
		                           TextureState{VK_IMAGE_USAGE_STORAGE_BIT},
		                           VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}});
		cmd_buffer.Bind(pso);
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, m_ggx_lut->GetView(view_desc))
		        .Bind(0, 1, m_charlie_lut->GetView(view_desc)));
		cmd_buffer.Dispatch(512 / 32, 512 / 32);
		cmd_buffer.Transition({},
		                      {TextureTransition{
		                           m_ggx_lut.get(),
		                           TextureState{VK_IMAGE_USAGE_STORAGE_BIT},
		                           TextureState{VK_IMAGE_USAGE_SAMPLED_BIT},
		                           VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
		                       TextureTransition{
		                           m_charlie_lut.get(),
		                           TextureState{VK_IMAGE_USAGE_STORAGE_BIT},
		                           TextureState{VK_IMAGE_USAGE_SAMPLED_BIT},
		                           VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}});
	}

	cmd_buffer.End();

	// Submit
	p_device->SubmitIdle(cmd_buffer);
}

void Renderer::MousePicking(uint32_t x, uint32_t y)
{
	if (!p_scene)
	{
		return;
	}

	auto &cmd_buffer = p_device->RequestCommandBuffer();
	if (!m_picking_buffer)
	{
		m_picking_buffer = std::make_unique<Buffer>(
		    p_device,
		    BufferDesc(sizeof(uint32_t), 1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU));
	}
}

void Renderer::DrawPrimitive()
{
	if (!p_depth_stencil || !p_present || !p_scene->GetRegistry().valid(p_scene->GetMainCamera()))
	{
		return;
	}

	auto &cmd_buffer = p_device->RequestCommandBuffer();
	cmd_buffer.Begin();
	cmd_buffer.Transition(
	    {},
	    {TextureTransition{
	         p_present,
	         TextureState{VK_IMAGE_USAGE_SAMPLED_BIT},
	         TextureState{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT},
	         VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
	     TextureTransition{
	         p_depth_stencil,
	         TextureState{VK_IMAGE_USAGE_SAMPLED_BIT},
	         TextureState{VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT},
	         VkImageSubresourceRange{VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1}}});

	DrawBVH(cmd_buffer);

	cmd_buffer.Transition(
	    {},
	    {TextureTransition{
	         p_present,
	         TextureState{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT},
	         TextureState{VK_IMAGE_USAGE_SAMPLED_BIT},
	         VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
	     TextureTransition{
	         p_depth_stencil,
	         TextureState{VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT},
	         TextureState{VK_IMAGE_USAGE_SAMPLED_BIT},
	         VkImageSubresourceRange{VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1}}});

	cmd_buffer.End();
	p_device->Submit(cmd_buffer);
}

void Renderer::DrawAABB(CommandBuffer &cmd_buffer, const std::vector<std::pair<AABB, glm::vec3>> &aabbs)
{
	const cmpt::Camera &main_camera = p_scene->GetRegistry().get<cmpt::Camera>(p_scene->GetMainCamera());

	ShaderDesc vertex_shader  = {};
	vertex_shader.filename    = "./Source/Shaders/DrawPrimitive.hlsl";
	vertex_shader.entry_point = "VSmain";
	vertex_shader.stage       = VK_SHADER_STAGE_VERTEX_BIT;
	vertex_shader.macros.push_back("DRAW_AABB");
	vertex_shader.type = ShaderType::HLSL;

	ShaderDesc fragment_shader  = {};
	fragment_shader.filename    = "./Source/Shaders/DrawPrimitive.hlsl";
	fragment_shader.entry_point = "PSmain";
	fragment_shader.stage       = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragment_shader.type        = ShaderType::HLSL;

	DynamicState dynamic_state = {};
	dynamic_state.dynamic_states.push_back(VK_DYNAMIC_STATE_SCISSOR);
	dynamic_state.dynamic_states.push_back(VK_DYNAMIC_STATE_VIEWPORT);

	ColorBlendState blend_state = {};
	blend_state.attachment_states.resize(1);
	blend_state.attachment_states.back().blend_enable = VK_FALSE;

	RasterizationState rasterization_state = {};
	rasterization_state.cull_mode          = VK_CULL_MODE_NONE;
	rasterization_state.polygon_mode       = VK_POLYGON_MODE_LINE;
	rasterization_state.line_width         = 3.f;

	InputAssemblyState input_assembly_state       = {};
	input_assembly_state.topology                 = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
	input_assembly_state.primitive_restart_enable = VK_TRUE;

	PipelineState pso;
	pso
	    .LoadShader(vertex_shader)
	    .LoadShader(fragment_shader)
	    .SetDynamicState(dynamic_state)
	    .SetColorBlendState(blend_state)
	    .SetInputAssemblyState(input_assembly_state)
	    .SetRasterizationState(rasterization_state);

	FrameBuffer framebuffer;
	framebuffer
	    .Bind(
	        p_present,
	        TextureViewDesc{
	            VK_IMAGE_VIEW_TYPE_2D,
	            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	        ColorAttachmentInfo{
	            /*samples				*/ VK_SAMPLE_COUNT_1_BIT,
	            /*load_op				*/ VK_ATTACHMENT_LOAD_OP_LOAD,
	            /*store_op			*/ VK_ATTACHMENT_STORE_OP_STORE,
	            /*stencil_load_op */ VK_ATTACHMENT_LOAD_OP_DONT_CARE,
	            /*stencil_store_op*/ VK_ATTACHMENT_STORE_OP_DONT_CARE,
	            /*clear_value    */ {}})
	    .Bind(
	        p_depth_stencil,
	        TextureViewDesc{
	            VK_IMAGE_VIEW_TYPE_2D,
	            VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1},
	        DepthStencilAttachmentInfo{
	            /*samples				*/ VK_SAMPLE_COUNT_1_BIT,
	            /*load_op				*/ VK_ATTACHMENT_LOAD_OP_LOAD,
	            /*store_op			*/ VK_ATTACHMENT_STORE_OP_STORE,
	            /*stencil_load_op */ VK_ATTACHMENT_LOAD_OP_DONT_CARE,
	            /*stencil_store_op*/ VK_ATTACHMENT_STORE_OP_DONT_CARE,
	            /*clear_value    */ {1.f, 0u}});

	struct
	{
		glm::mat4 view_projection      = {};
		alignas(16) glm::vec3 aabb_min = {};
		alignas(16) glm::vec3 aabb_max = {};
		alignas(16) glm::vec3 color    = {};
	} push_constants;

	push_constants.view_projection = main_camera.GetViewProjection();

	// Record command buffer
	{
		cmd_buffer.SetViewport(static_cast<float>(m_extent.width), -static_cast<float>(m_extent.height), 0, static_cast<float>(m_extent.height));
		cmd_buffer.SetScissor(m_extent.width, m_extent.height);

		cmd_buffer.BeginRenderPass(framebuffer);
		cmd_buffer.Bind(pso);
		for (const auto &[aabb, color] : aabbs)
		{
			push_constants.aabb_max = aabb.GetMax();
			push_constants.aabb_min = aabb.GetMin();
			push_constants.color    = color;

			cmd_buffer.PushConstants(VK_SHADER_STAGE_VERTEX_BIT, &push_constants, sizeof(push_constants), 0);
			cmd_buffer.Draw(16);
		}

		cmd_buffer.EndRenderPass();
	}
}

void Renderer::DrawBVH(CommandBuffer &cmd_buffer)
{
	const cmpt::Camera &main_camera = p_scene->GetRegistry().get<cmpt::Camera>(p_scene->GetMainCamera());

	auto group = p_scene->GetRegistry().group<cmpt::MeshRenderer>(entt::get<cmpt::Transform>);

	if (group.empty())
	{
		return;
	}

	ShaderDesc vertex_shader  = {};
	vertex_shader.filename    = "./Source/Shaders/BVH/VisualizeBVH_.hlsl";
	vertex_shader.entry_point = "VSmain";
	vertex_shader.stage       = VK_SHADER_STAGE_VERTEX_BIT;
	vertex_shader.type        = ShaderType::HLSL;

	ShaderDesc geometry_shader  = {};
	geometry_shader.filename    = "./Source/Shaders/BVH/VisualizeBVH_.hlsl";
	geometry_shader.entry_point = "GSmain";
	geometry_shader.stage       = VK_SHADER_STAGE_GEOMETRY_BIT;
	geometry_shader.type        = ShaderType::HLSL;

	ShaderDesc fragment_shader  = {};
	fragment_shader.filename    = "./Source/Shaders/BVH/VisualizeBVH_.hlsl";
	fragment_shader.entry_point = "PSmain";
	fragment_shader.stage       = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragment_shader.type        = ShaderType::HLSL;

	DynamicState dynamic_state = {};
	dynamic_state.dynamic_states.push_back(VK_DYNAMIC_STATE_SCISSOR);
	dynamic_state.dynamic_states.push_back(VK_DYNAMIC_STATE_VIEWPORT);

	ColorBlendState blend_state = {};
	blend_state.attachment_states.resize(1);
	blend_state.attachment_states.back().blend_enable = VK_FALSE;

	RasterizationState rasterization_state = {};
	rasterization_state.cull_mode          = VK_CULL_MODE_NONE;
	rasterization_state.polygon_mode       = VK_POLYGON_MODE_LINE;
	rasterization_state.line_width         = 1.f;

	InputAssemblyState input_assembly_state       = {};
	input_assembly_state.topology                 = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
	input_assembly_state.primitive_restart_enable = VK_TRUE;

	PipelineState pso;
	pso
	    .LoadShader(vertex_shader)
	    .LoadShader(geometry_shader)
	    .LoadShader(fragment_shader)
	    .SetDynamicState(dynamic_state)
	    .SetColorBlendState(blend_state)
	    .SetInputAssemblyState(input_assembly_state)
	    .SetRasterizationState(rasterization_state);

	FrameBuffer framebuffer;
	framebuffer
	    .Bind(
	        p_present,
	        TextureViewDesc{
	            VK_IMAGE_VIEW_TYPE_2D,
	            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	        ColorAttachmentInfo{
	            /*samples				*/ VK_SAMPLE_COUNT_1_BIT,
	            /*load_op				*/ VK_ATTACHMENT_LOAD_OP_LOAD,
	            /*store_op			*/ VK_ATTACHMENT_STORE_OP_STORE,
	            /*stencil_load_op */ VK_ATTACHMENT_LOAD_OP_DONT_CARE,
	            /*stencil_store_op*/ VK_ATTACHMENT_STORE_OP_DONT_CARE,
	            /*clear_value    */ {}})
	    .Bind(
	        p_depth_stencil,
	        TextureViewDesc{
	            VK_IMAGE_VIEW_TYPE_2D,
	            VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1},
	        DepthStencilAttachmentInfo{
	            /*samples				*/ VK_SAMPLE_COUNT_1_BIT,
	            /*load_op				*/ VK_ATTACHMENT_LOAD_OP_LOAD,
	            /*store_op			*/ VK_ATTACHMENT_STORE_OP_STORE,
	            /*stencil_load_op */ VK_ATTACHMENT_LOAD_OP_DONT_CARE,
	            /*stencil_store_op*/ VK_ATTACHMENT_STORE_OP_DONT_CARE,
	            /*clear_value    */ {1.f, 0u}});

	struct
	{
		glm::mat4 view_projection = {};
		glm::mat4 transform       = {};
		uint32_t  instance_id     = 0;
	} push_constants;

	push_constants.view_projection = main_camera.GetViewProjection();

	// Record command buffer
	{
		cmd_buffer.SetViewport(static_cast<float>(m_extent.width), -static_cast<float>(m_extent.height), 0, static_cast<float>(m_extent.height));
		cmd_buffer.SetScissor(m_extent.width, m_extent.height);

		cmd_buffer.BeginRenderPass(framebuffer);
		cmd_buffer.Bind(pso);

		std::vector<Buffer *> hierarchy_buffer;
		std::vector<Buffer *> aabbs_buffer;
		std::vector<uint32_t> volume_count;
		hierarchy_buffer.reserve(group.size());
		aabbs_buffer.reserve(group.size());
		volume_count.reserve(group.size());

		group.each([&](cmpt::MeshRenderer &mesh_renderer, cmpt::Transform &transform) {
			auto *mesh = mesh_renderer.GetMesh();
			if (mesh)
			{
				hierarchy_buffer.push_back(&mesh->GetBLAS().GetHierarchyBuffer());
				aabbs_buffer.push_back(&mesh->GetBLAS().GetBoundingVolumeBuffer());
				volume_count.push_back(mesh->GetIndicesCount() / 3 * 2 - 1);
			}
		});

		cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
		                    .Bind(0, 0, hierarchy_buffer)
		                    .Bind(0, 1, aabbs_buffer));

		group.each([&](cmpt::MeshRenderer &mesh_renderer, cmpt::Transform &transform) {
			push_constants.transform = transform.GetWorldTransform();
			cmd_buffer.PushConstants(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT, &push_constants, sizeof(push_constants), 0);
			cmd_buffer.Draw((volume_count[push_constants.instance_id] + 1) / 2, 1, (volume_count[push_constants.instance_id] + 1) / 2-1);
			push_constants.instance_id++;
		});

		cmd_buffer.EndRenderPass();
	}
}
}        // namespace Ilum