#include "Light.hpp"
#include "Camera.hpp"
#include "Transform.hpp"

#include "Scene/Entity.hpp"

#include <RHI/Command.hpp>
#include <RHI/DescriptorState.hpp>
#include <RHI/FrameBuffer.hpp>
#include <RHI/PipelineState.hpp>
#include <RHI/Texture.hpp>

#include <Asset/AssetManager.hpp>

#include <Shaders/ShaderInterop.hpp>

#include <glm/gtc/type_ptr.hpp>

namespace Ilum::cmpt
{
bool Light::OnImGui(ImGuiContext &context)
{
	const char *const light_type[] = {"Point", "Directional", "Spot", "Area"};
	m_update |= ImGui::Combo("Type", reinterpret_cast<int32_t *>(&m_type), light_type, 4);

	m_update |= ImGui::ColorEdit3("Color", glm::value_ptr(m_color));
	m_update |= ImGui::DragFloat("Intensity", &m_intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	if (m_type == LightType::Point)
	{
		m_update |= ImGui::DragFloat("Range", &m_range, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
	}
	else if (m_type == LightType::Spot)
	{
		m_update |= ImGui::DragFloat("Range", &m_range, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
		m_update |= ImGui::DragFloat("Cut off", &m_spot_inner_cone_angle, 0.0001f, 0.f, 1.f, "%.5f");
		m_update |= ImGui::DragFloat("Outer cut off", &m_spot_outer_cone_angle, 0.0001f, 0.f, m_spot_inner_cone_angle, "%.5f");
		if (ImGui::TreeNode("Shadowmap"))
		{
			ImGui::Image(context.TextureID(m_shadow_map->GetView(TextureViewDesc{VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1})), ImVec2(300, 300));
			ImGui::TreePop();
		}
	}
	else if (m_type == LightType::Area)
	{
		const char *const light_shape[] = {"Rectangle", "Ellipse"};
		m_update |= ImGui::Combo("Shape", reinterpret_cast<int32_t *>(&m_shape), light_shape, 2);
	}

	return m_update;
}

void Light::Tick(Scene &scene, entt::entity entity, RHIDevice *device)
{
	if (m_update)
	{
		switch (m_type)
		{
			case cmpt::LightType::Point:
				UpdatePointLight(scene, entity, device);
				break;
			case cmpt::LightType::Directional:
				UpdateDirectionalLight(scene, entity, device);
				break;
			case cmpt::LightType::Spot:
				UpdateSpotLight(scene, entity, device);
				break;
			case cmpt::LightType::Area:
				UpdateAreaLight(scene, entity, device);
				break;
			default:
				break;
		}
		m_update = false;
	}
}

Texture *Light::GetShadowmap()
{
	if (m_shadow_map)
	{
		return m_shadow_map.get();
	}
	return nullptr;
}

Buffer *Light::GetBuffer()
{
	if (m_buffer)
	{
		return m_buffer.get();
	}
	return nullptr;
}

void Light::SetType(LightType type)
{
	m_type = type;
	Update();
}

LightType Light::GetType() const
{
	return m_type;
}

void Light::UpdateDirectionalLight(Scene &scene, entt::entity entity, RHIDevice *device)
{
	// Create GPU Resource
	if (!m_buffer || m_buffer->GetSize() != sizeof(ShaderInterop::DirectionalLight))
	{
		m_buffer = std::make_unique<Buffer>(device, BufferDesc(sizeof(ShaderInterop::DirectionalLight), 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU));
	}
	if (!m_shadow_map || m_shadow_map->GetLayerCount() != 4)
	{
		m_shadow_map = std::make_unique<Texture>(
		    device, TextureDesc{
		                1024,
		                1024,
		                1,
		                1,
		                4,
		                VK_SAMPLE_COUNT_1_BIT,
		                VK_FORMAT_D32_SFLOAT,
		                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT});

		auto &cmd_buffer = device->RequestCommandBuffer();
		cmd_buffer.Begin();
		cmd_buffer.Transition(
		    m_shadow_map.get(),
		    TextureState{},
		    TextureState(VK_IMAGE_USAGE_SAMPLED_BIT),
		    VkImageSubresourceRange{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 4});
		cmd_buffer.End();
		device->SubmitIdle(cmd_buffer);
	}

	std::array<glm::mat4, 4> shadow_matrics = {};
	std::array<glm::vec3, 4> shadow_cam_pos = {};
	struct
	{
		glm::mat4 view_projection = {};
		glm::vec3 position        = {};
		uint32_t  cascade_idx     = {};
		uint32_t  instance_id     = {};
		uint32_t  meshlet_count   = {};
	} push_data;

	// Update Buffer
	{
		ShaderInterop::DirectionalLight *data = static_cast<ShaderInterop::DirectionalLight *>(m_buffer->Map());
		data->color                           = m_color;
		data->intensity                       = m_intensity;
		// Update direction
		{
			Entity e         = Entity(scene, entity);
			auto  &transform = e.GetComponent<cmpt::Transform>();
			data->direction  = glm::mat3_cast(glm::qua<float>(glm::radians(transform.GetRotation()))) * glm::vec3(0.f, -1.f, 0.f);
		}
		// Update cascade shadow
		{
			Entity camera_entity = Entity(scene, scene.GetMainCamera());
			if (camera_entity.IsValid() && camera_entity.HasComponent<cmpt::Camera>())
			{
				const auto &camera            = camera_entity.GetComponent<cmpt::Camera>();
				float       cascade_splits[4] = {0.f};

				float near_clip  = camera.GetNearPlane();
				float far_clip   = camera.GetFarPlane();
				float clip_range = far_clip - near_clip;
				float ratio      = far_clip / near_clip;

				// Calculate split depths based on view camera frustum
				for (uint32_t i = 0; i < 4; i++)
				{
					float p           = (static_cast<float>(i) + 1.f) / 4.f;
					float log         = near_clip * std::pow(ratio, p);
					float uniform     = near_clip + clip_range * p;
					float d           = 0.95f * (log - uniform) + uniform;
					cascade_splits[i] = (d - near_clip) / clip_range;
				}

				// Calculate orthographic projection matrix for each cascade
				float last_split_dist = 0.f;
				for (uint32_t i = 0; i < 4; i++)
				{
					float split_dist = cascade_splits[i];

					glm::vec3 frustum_corners[8] = {
					    glm::vec3(-1.0f, 1.0f, 0.0f),
					    glm::vec3(1.0f, 1.0f, 0.0f),
					    glm::vec3(1.0f, -1.0f, 0.0f),
					    glm::vec3(-1.0f, -1.0f, 0.0f),
					    glm::vec3(-1.0f, 1.0f, 1.0f),
					    glm::vec3(1.0f, 1.0f, 1.0f),
					    glm::vec3(1.0f, -1.0f, 1.0f),
					    glm::vec3(-1.0f, -1.0f, 1.0f)};

					// Project frustum corners into world space
					glm::mat4 inv_cam = glm::inverse(camera.GetViewProjection());
					for (uint32_t j = 0; j < 8; j++)
					{
						glm::vec4 inv_corner = inv_cam * glm::vec4(frustum_corners[j], 1.f);
						frustum_corners[j]   = glm::vec3(inv_corner / inv_corner.w);
					}

					for (uint32_t j = 0; j < 4; j++)
					{
						glm::vec3 corner_ray   = frustum_corners[j + 4] - frustum_corners[j];
						frustum_corners[j + 4] = frustum_corners[j] + corner_ray * split_dist;
						frustum_corners[j]     = frustum_corners[j] + corner_ray * last_split_dist;
					}

					// Get frustum center
					glm::vec3 frustum_center = glm::vec3(0.0f);
					for (uint32_t j = 0; j < 8; j++)
					{
						frustum_center += frustum_corners[j];
					}
					frustum_center /= 8.0f;

					float radius = 0.0f;
					for (uint32_t j = 0; j < 8; j++)
					{
						float distance = glm::length(frustum_corners[j] - frustum_center);
						radius         = glm::max(radius, distance);
					}
					radius = std::ceil(radius * 16.0f) / 16.0f;

					glm::vec3 max_extents = glm::vec3(radius);
					glm::vec3 min_extents = -max_extents;

					glm::vec3 light_dir = glm::normalize(data->direction);

					data->shadow_cam_pos[i] = glm::vec4(frustum_center - light_dir * max_extents.z, 1.0);

					glm::mat4 light_view_matrix  = glm::lookAt(glm::vec3(data->shadow_cam_pos[i]), frustum_center, glm::vec3(0.0f, 1.0f, 0.0f));
					glm::mat4 light_ortho_matrix = glm::ortho(min_extents.x, max_extents.x, min_extents.y, max_extents.y, -2.f * (max_extents.z - min_extents.z), max_extents.z - min_extents.z);

					// Store split distance and matrix in cascade
					data->split_depth[i]     = -(near_clip + split_dist * clip_range);
					data->view_projection[i] = light_ortho_matrix * light_view_matrix;

					// Stablize
					glm::vec3 shadow_origin = glm::vec3(0.0f);
					shadow_origin           = (data->view_projection[i] * glm::vec4(shadow_origin, 1.0f));
					shadow_origin *= 1024.f;

					glm::vec3 rounded_origin = glm::round(shadow_origin);
					glm::vec3 round_offset   = rounded_origin - shadow_origin;
					round_offset             = round_offset / 1024.f;
					round_offset.z           = 0.0f;

					data->view_projection[i][3][0] += round_offset.x;
					data->view_projection[i][3][1] += round_offset.y;

					last_split_dist = cascade_splits[i];

					shadow_matrics[i] = data->view_projection[i];
					shadow_cam_pos[i] = data->shadow_cam_pos[i];
				}
			}
		}

		m_buffer->Flush(m_buffer->GetSize());
		m_buffer->Unmap();
	}

	// Render Cascade Shadowmap
	{
		DynamicState dynamic_state;
		dynamic_state.dynamic_states = {
		    VK_DYNAMIC_STATE_VIEWPORT,
		    VK_DYNAMIC_STATE_SCISSOR};

		ShaderDesc task_shader  = {};
		task_shader.filename    = "./Source/Shaders/Shadow/CascadeShadowmap.hlsl";
		task_shader.entry_point = "ASmain";
		task_shader.stage       = VK_SHADER_STAGE_TASK_BIT_NV;
		task_shader.type        = ShaderType::HLSL;

		ShaderDesc mesh_shader  = {};
		mesh_shader.filename    = "./Source/Shaders/Shadow/CascadeShadowmap.hlsl";
		mesh_shader.entry_point = "MSmain";
		mesh_shader.stage       = VK_SHADER_STAGE_MESH_BIT_NV;
		mesh_shader.type        = ShaderType::HLSL;

		ShaderDesc opaque_frag_shader  = {};
		opaque_frag_shader.filename    = "./Source/Shaders/Shadow/CascadeShadowmap.hlsl";
		opaque_frag_shader.entry_point = "PSmain";
		opaque_frag_shader.stage       = VK_SHADER_STAGE_FRAGMENT_BIT;
		opaque_frag_shader.type        = ShaderType::HLSL;

		ShaderDesc alpha_frag_shader  = {};
		alpha_frag_shader.filename    = "./Source/Shaders/Shadow/CascadeShadowmap.hlsl";
		alpha_frag_shader.entry_point = "PSmain";
		alpha_frag_shader.stage       = VK_SHADER_STAGE_FRAGMENT_BIT;
		alpha_frag_shader.type        = ShaderType::HLSL;
		alpha_frag_shader.macros.push_back("ALPHA_TEST");

		PipelineState opaque_pso;
		opaque_pso
		    .SetName("CascadeShadowmap - Opaque")
		    .SetDynamicState(dynamic_state)
		    .LoadShader(task_shader)
		    .LoadShader(mesh_shader)
		    .LoadShader(opaque_frag_shader);

		PipelineState alpha_pso;
		alpha_pso
		    .SetName("CascadeShadowmap - Alpha")
		    .SetDynamicState(dynamic_state)
		    .LoadShader(task_shader)
		    .LoadShader(mesh_shader)
		    .LoadShader(alpha_frag_shader);

		FrameBuffer                framebuffer;
		DepthStencilAttachmentInfo attachment_info = {};

		framebuffer.Bind(
		    m_shadow_map.get(),
		    TextureViewDesc{VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 4},
		    attachment_info);

		auto &cmd_buffer = device->RequestCommandBuffer();
		cmd_buffer.Begin();
		cmd_buffer.Transition(m_shadow_map.get(), TextureState{VK_IMAGE_USAGE_SAMPLED_BIT}, TextureState{VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT}, VkImageSubresourceRange{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 4});
		cmd_buffer.BeginRenderPass(framebuffer);

		cmd_buffer.SetViewport(static_cast<float>(m_shadow_map->GetWidth()), -static_cast<float>(m_shadow_map->GetHeight()), 0, static_cast<float>(m_shadow_map->GetHeight()));
		cmd_buffer.SetScissor(m_shadow_map->GetWidth(), m_shadow_map->GetHeight());

		// Draw Opaque
		{
			auto batch = scene.Batch(AlphaMode::Opaque);

			std::vector<Buffer *> instances;
			instances.reserve(batch.meshes.size());
			for (auto &mesh : batch.meshes)
			{
				instances.push_back(mesh->GetBuffer());
			}

			if (!instances.empty())
			{
				cmd_buffer.Bind(opaque_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, instances)
				                    .Bind(0, 1, scene.GetAssetManager().GetMeshletBuffer())
				                    .Bind(0, 2, scene.GetAssetManager().GetVertexBuffer())
				                    .Bind(0, 3, scene.GetAssetManager().GetMeshletVertexBuffer())
				                    .Bind(0, 4, scene.GetAssetManager().GetMeshletTriangleBuffer()));

				for (push_data.cascade_idx = 0; push_data.cascade_idx < 4; push_data.cascade_idx++)
				{
					push_data.view_projection = shadow_matrics[push_data.cascade_idx];
					push_data.position        = shadow_cam_pos[push_data.cascade_idx];

					uint32_t instance_id = 0;
					for (auto &mesh : batch.meshes)
					{
						push_data.instance_id   = instance_id;
						push_data.meshlet_count = mesh->GetMesh()->GetMeshletsCount();
						cmd_buffer.PushConstants(VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_TASK_BIT_NV, &push_data, sizeof(push_data), 0);
						vkCmdDrawMeshTasksNV(cmd_buffer, (push_data.meshlet_count + 32 - 1) / 32, 0);
						instance_id++;
					}
				}
			}
		}

		// Draw Alpha
		{
			auto batch = scene.Batch(AlphaMode::Masked | AlphaMode::Blend);

			std::vector<Buffer *> instances;
			instances.reserve(batch.meshes.size());
			for (auto &mesh : batch.meshes)
			{
				instances.push_back(mesh->GetBuffer());
			}

			if (!instances.empty())
			{
				cmd_buffer.Bind(alpha_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, instances)
				                    .Bind(0, 1, scene.GetAssetManager().GetMeshletBuffer())
				                    .Bind(0, 2, scene.GetAssetManager().GetVertexBuffer())
				                    .Bind(0, 3, scene.GetAssetManager().GetMeshletVertexBuffer())
				                    .Bind(0, 4, scene.GetAssetManager().GetMeshletTriangleBuffer())
				                    .Bind(0, 5, scene.GetAssetManager().GetMaterialBuffer())
				                    .Bind(0, 6, scene.GetAssetManager().GetTextureViews())
				                    .Bind(0, 7, device->AllocateSampler(SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_LINEAR})));

				for (push_data.cascade_idx = 0; push_data.cascade_idx < 4; push_data.cascade_idx++)
				{
					push_data.view_projection = shadow_matrics[push_data.cascade_idx];
					push_data.position        = shadow_cam_pos[push_data.cascade_idx];

					uint32_t instance_id = 0;
					for (uint32_t i = 0; i < batch.order.size(); i++)
					{
						push_data.instance_id   = instance_id;
						push_data.meshlet_count = batch.meshes[i]->GetMesh()->GetMeshletsCount();
						cmd_buffer.PushConstants(VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_TASK_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT, &push_data, sizeof(push_data), 0);
						vkCmdDrawMeshTasksNV(cmd_buffer, (push_data.meshlet_count + 32 - 1) / 32, 0);
					}
				}
			}
		}

		cmd_buffer.EndRenderPass();
		cmd_buffer.Transition(m_shadow_map.get(), TextureState{VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT}, TextureState{VK_IMAGE_USAGE_SAMPLED_BIT}, VkImageSubresourceRange{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 4});
		cmd_buffer.End();
		device->Submit(cmd_buffer);
	}
}

void Light::UpdatePointLight(Scene &scene, entt::entity entity, RHIDevice *device)
{
	// Create GPU resource
	if (!m_buffer || m_buffer->GetSize() != sizeof(ShaderInterop::PointLight))
	{
		m_buffer = std::make_unique<Buffer>(device, BufferDesc(sizeof(ShaderInterop::PointLight), 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU));
	}
	if (!m_shadow_map || m_shadow_map->GetLayerCount() != 6)
	{
		m_shadow_map = std::make_unique<Texture>(
		    device, TextureDesc{
		                1024,
		                1024,
		                1,
		                1,
		                6,
		                VK_SAMPLE_COUNT_1_BIT,
		                VK_FORMAT_D32_SFLOAT,
		                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT});

		auto &cmd_buffer = device->RequestCommandBuffer();
		cmd_buffer.Begin();
		cmd_buffer.Transition(
		    m_shadow_map.get(),
		    TextureState{},
		    TextureState(VK_IMAGE_USAGE_SAMPLED_BIT),
		    VkImageSubresourceRange{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6});
		cmd_buffer.End();
		device->SubmitIdle(cmd_buffer);
	}

	struct
	{
		glm::mat4 view_projection = {};
		glm::vec3 position        = {};
		uint32_t  face_id         = {};
		uint32_t  instance_id     = {};
		uint32_t  meshlet_count   = {};
	} push_data;

	// Update Buffer
	{
		Entity e = Entity(scene, entity);

		auto &transform = e.GetComponent<cmpt::Transform>();

		ShaderInterop::PointLight *data = static_cast<ShaderInterop::PointLight *>(m_buffer->Map());

		data->color     = m_color;
		data->intensity = m_intensity;
		data->position  = transform.GetWorldTransform()[3];
		data->range     = m_range;

		push_data.position = data->position;

		m_buffer->Flush(m_buffer->GetSize());
		m_buffer->Unmap();
	}

	// Render Onmidirectional Shadow Map
	{
		glm::mat4 projection_matrix = glm::perspective(glm::radians(90.0f), 1.0f, 0.01f, 100.f);

		std::array<glm::mat4, 6> shadow_matrics = {
		    projection_matrix * glm::lookAt(push_data.position, push_data.position + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
		    projection_matrix * glm::lookAt(push_data.position, push_data.position + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
		    projection_matrix * glm::lookAt(push_data.position, push_data.position + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
		    projection_matrix * glm::lookAt(push_data.position, push_data.position + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
		    projection_matrix * glm::lookAt(push_data.position, push_data.position + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
		    projection_matrix * glm::lookAt(push_data.position, push_data.position + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f))};

		DynamicState dynamic_state;
		dynamic_state.dynamic_states = {
		    VK_DYNAMIC_STATE_VIEWPORT,
		    VK_DYNAMIC_STATE_SCISSOR};

		ShaderDesc task_shader  = {};
		task_shader.filename    = "./Source/Shaders/Shadow/OnmiShadowmap.hlsl";
		task_shader.entry_point = "ASmain";
		task_shader.stage       = VK_SHADER_STAGE_TASK_BIT_NV;
		task_shader.type        = ShaderType::HLSL;

		ShaderDesc mesh_shader  = {};
		mesh_shader.filename    = "./Source/Shaders/Shadow/OnmiShadowmap.hlsl";
		mesh_shader.entry_point = "MSmain";
		mesh_shader.stage       = VK_SHADER_STAGE_MESH_BIT_NV;
		mesh_shader.type        = ShaderType::HLSL;

		ShaderDesc opaque_frag_shader  = {};
		opaque_frag_shader.filename    = "./Source/Shaders/Shadow/OnmiShadowmap.hlsl";
		opaque_frag_shader.entry_point = "PSmain";
		opaque_frag_shader.stage       = VK_SHADER_STAGE_FRAGMENT_BIT;
		opaque_frag_shader.type        = ShaderType::HLSL;

		ShaderDesc alpha_frag_shader  = {};
		alpha_frag_shader.filename    = "./Source/Shaders/Shadow/OnmiShadowmap.hlsl";
		alpha_frag_shader.entry_point = "PSmain";
		alpha_frag_shader.stage       = VK_SHADER_STAGE_FRAGMENT_BIT;
		alpha_frag_shader.type        = ShaderType::HLSL;
		alpha_frag_shader.macros.push_back("ALPHA_TEST");

		PipelineState opaque_pso;
		opaque_pso
		    .SetName("OnmiShadowmap - Opaque")
		    .SetDynamicState(dynamic_state)
		    .LoadShader(task_shader)
		    .LoadShader(mesh_shader)
		    .LoadShader(opaque_frag_shader);

		PipelineState alpha_pso;
		alpha_pso
		    .SetName("OnmiShadowmap - Alpha")
		    .SetDynamicState(dynamic_state)
		    .LoadShader(task_shader)
		    .LoadShader(mesh_shader)
		    .LoadShader(alpha_frag_shader);

		FrameBuffer                framebuffer;
		DepthStencilAttachmentInfo attachment_info = {};

		framebuffer.Bind(
		    m_shadow_map.get(),
		    TextureViewDesc{VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6},
		    attachment_info);

		auto &cmd_buffer = device->RequestCommandBuffer();
		cmd_buffer.Begin();
		cmd_buffer.Transition(m_shadow_map.get(), TextureState{VK_IMAGE_USAGE_SAMPLED_BIT}, TextureState{VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT}, VkImageSubresourceRange{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6});
		cmd_buffer.BeginRenderPass(framebuffer);

		cmd_buffer.SetViewport(static_cast<float>(m_shadow_map->GetWidth()), -static_cast<float>(m_shadow_map->GetHeight()), 0, static_cast<float>(m_shadow_map->GetHeight()));
		cmd_buffer.SetScissor(m_shadow_map->GetWidth(), m_shadow_map->GetHeight());

		// Draw Opaque
		{
			auto batch = scene.Batch(AlphaMode::Opaque);

			std::vector<Buffer *> instances;
			instances.reserve(batch.meshes.size());
			for (auto &mesh : batch.meshes)
			{
				instances.push_back(mesh->GetBuffer());
			}

			if (!instances.empty())
			{
				cmd_buffer.Bind(opaque_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, instances)
				                    .Bind(0, 1, scene.GetAssetManager().GetMeshletBuffer())
				                    .Bind(0, 2, scene.GetAssetManager().GetVertexBuffer())
				                    .Bind(0, 3, scene.GetAssetManager().GetMeshletVertexBuffer())
				                    .Bind(0, 4, scene.GetAssetManager().GetMeshletTriangleBuffer()));

				for (push_data.face_id = 0; push_data.face_id < 6; push_data.face_id++)
				{
					push_data.view_projection = shadow_matrics[push_data.face_id];

					uint32_t instance_id = 0;
					for (auto &mesh : batch.meshes)
					{
						push_data.instance_id   = instance_id;
						push_data.meshlet_count = mesh->GetMesh()->GetMeshletsCount();
						cmd_buffer.PushConstants(VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_TASK_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT, &push_data, sizeof(push_data), 0);
						vkCmdDrawMeshTasksNV(cmd_buffer, (push_data.meshlet_count + 32 - 1) / 32, 0);
						instance_id++;
					}
				}
			}
		}

		// Draw Alpha
		{
			auto batch = scene.Batch(AlphaMode::Masked | AlphaMode::Blend);

			std::vector<Buffer *> instances;
			instances.reserve(batch.meshes.size());
			for (auto &mesh : batch.meshes)
			{
				instances.push_back(mesh->GetBuffer());
			}

			if (!instances.empty())
			{
				cmd_buffer.Bind(alpha_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, instances)
				                    .Bind(0, 1, scene.GetAssetManager().GetMeshletBuffer())
				                    .Bind(0, 2, scene.GetAssetManager().GetVertexBuffer())
				                    .Bind(0, 3, scene.GetAssetManager().GetMeshletVertexBuffer())
				                    .Bind(0, 4, scene.GetAssetManager().GetMeshletTriangleBuffer())
				                    .Bind(0, 5, scene.GetAssetManager().GetMaterialBuffer())
				                    .Bind(0, 6, scene.GetAssetManager().GetTextureViews())
				                    .Bind(0, 7, device->AllocateSampler(SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_LINEAR})));

				for (push_data.face_id = 0; push_data.face_id < 6; push_data.face_id++)
				{
					push_data.view_projection = shadow_matrics[push_data.face_id];

					uint32_t instance_id = 0;
					for (uint32_t i = 0; i < batch.order.size(); i++)
					{
						push_data.instance_id   = instance_id;
						push_data.meshlet_count = batch.meshes[i]->GetMesh()->GetMeshletsCount();
						cmd_buffer.PushConstants(VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_TASK_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT, &push_data, sizeof(push_data), 0);
						vkCmdDrawMeshTasksNV(cmd_buffer, (push_data.meshlet_count + 32 - 1) / 32, 0);
					}
				}
			}
		}

		cmd_buffer.EndRenderPass();
		cmd_buffer.Transition(m_shadow_map.get(), TextureState{VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT}, TextureState{VK_IMAGE_USAGE_SAMPLED_BIT}, VkImageSubresourceRange{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6});
		cmd_buffer.End();
		device->Submit(cmd_buffer);
	}
}

void Light::UpdateSpotLight(Scene &scene, entt::entity entity, RHIDevice *device)
{
	// Create GPU resource
	if (!m_buffer || m_buffer->GetSize() != sizeof(ShaderInterop::SpotLight))
	{
		m_buffer = std::make_unique<Buffer>(device, BufferDesc(sizeof(ShaderInterop::SpotLight), 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU));
	}
	if (!m_shadow_map || m_shadow_map->GetLayerCount() != 1)
	{
		m_shadow_map = std::make_unique<Texture>(
		    device, TextureDesc{
		                1024,
		                1024,
		                1,
		                1,
		                1,
		                VK_SAMPLE_COUNT_1_BIT,
		                VK_FORMAT_D32_SFLOAT,
		                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT});

		auto &cmd_buffer = device->RequestCommandBuffer();
		cmd_buffer.Begin();
		cmd_buffer.Transition(
		    m_shadow_map.get(),
		    TextureState{},
		    TextureState(VK_IMAGE_USAGE_SAMPLED_BIT),
		    VkImageSubresourceRange{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1});
		cmd_buffer.End();
		device->SubmitIdle(cmd_buffer);
	}

	struct
	{
		glm::mat4 view_projection = {};
		glm::vec3 position        = {};
		uint32_t  instance_id     = {};
		uint32_t  meshlet_count   = {};
	} push_data;

	// Update Buffer
	{
		Entity e = Entity(scene, entity);

		auto &transform = e.GetComponent<cmpt::Transform>();

		ShaderInterop::SpotLight *data = static_cast<ShaderInterop::SpotLight *>(m_buffer->Map());

		data->color           = m_color;
		data->intensity       = m_intensity;
		data->cut_off         = glm::radians(m_spot_inner_cone_angle);
		data->outer_cut_off   = glm::radians(m_spot_outer_cone_angle);
		data->position        = transform.GetWorldTransform()[3];
		data->direction       = glm::mat3_cast(glm::qua<float>(glm::radians(transform.GetRotation()))) * glm::vec3(0.f, -1.f, 0.f);
		data->view_projection = glm::perspective(2.f * m_spot_outer_cone_angle, 1.0f, 0.01f, 1000.f) * glm::lookAt(transform.GetTranslation(), transform.GetTranslation() + data->direction, glm::vec3(0.f, 1.f, 0.f));

		push_data.view_projection = data->view_projection;
		push_data.position        = data->position;

		m_buffer->Flush(m_buffer->GetSize());
		m_buffer->Unmap();
	}

	// Render Shadow Map
	{
		DynamicState dynamic_state;
		dynamic_state.dynamic_states = {
		    VK_DYNAMIC_STATE_VIEWPORT,
		    VK_DYNAMIC_STATE_SCISSOR,
		    VK_DYNAMIC_STATE_DEPTH_BIAS};

		ShaderDesc task_shader  = {};
		task_shader.filename    = "./Source/Shaders/Shadow/Shadowmap.hlsl";
		task_shader.entry_point = "ASmain";
		task_shader.stage       = VK_SHADER_STAGE_TASK_BIT_NV;
		task_shader.type        = ShaderType::HLSL;

		ShaderDesc mesh_shader  = {};
		mesh_shader.filename    = "./Source/Shaders/Shadow/Shadowmap.hlsl";
		mesh_shader.entry_point = "MSmain";
		mesh_shader.stage       = VK_SHADER_STAGE_MESH_BIT_NV;
		mesh_shader.type        = ShaderType::HLSL;

		ShaderDesc opaque_frag_shader  = {};
		opaque_frag_shader.filename    = "./Source/Shaders/Shadow/Shadowmap.hlsl";
		opaque_frag_shader.entry_point = "PSmain";
		opaque_frag_shader.stage       = VK_SHADER_STAGE_FRAGMENT_BIT;
		opaque_frag_shader.type        = ShaderType::HLSL;

		ShaderDesc alpha_frag_shader  = {};
		alpha_frag_shader.filename    = "./Source/Shaders/Shadow/Shadowmap.hlsl";
		alpha_frag_shader.entry_point = "PSmain";
		alpha_frag_shader.stage       = VK_SHADER_STAGE_FRAGMENT_BIT;
		alpha_frag_shader.type        = ShaderType::HLSL;
		alpha_frag_shader.macros.push_back("ALPHA_TEST");

		PipelineState opaque_pso;
		opaque_pso
		    .SetName("Shadowmap - Opaque")
		    .SetDynamicState(dynamic_state)
		    .LoadShader(task_shader)
		    .LoadShader(mesh_shader)
		    .LoadShader(opaque_frag_shader);

		PipelineState alpha_pso;
		alpha_pso
		    .SetName("Shadowmap - Alpha")
		    .SetDynamicState(dynamic_state)
		    .LoadShader(task_shader)
		    .LoadShader(mesh_shader)
		    .LoadShader(alpha_frag_shader);

		FrameBuffer                framebuffer;
		DepthStencilAttachmentInfo attachment_info = {};

		framebuffer.Bind(
		    m_shadow_map.get(),
		    TextureViewDesc{VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
		    attachment_info);

		auto &cmd_buffer = device->RequestCommandBuffer();
		cmd_buffer.Begin();
		cmd_buffer.Transition(m_shadow_map.get(), TextureState{VK_IMAGE_USAGE_SAMPLED_BIT}, TextureState{VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT}, VkImageSubresourceRange{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1});
		cmd_buffer.BeginRenderPass(framebuffer);

		cmd_buffer.SetViewport(static_cast<float>(m_shadow_map->GetWidth()), -static_cast<float>(m_shadow_map->GetHeight()), 0, static_cast<float>(m_shadow_map->GetHeight()));
		cmd_buffer.SetScissor(m_shadow_map->GetWidth(), m_shadow_map->GetHeight());
		cmd_buffer.SetDepthBias(4.f, 0.f, 1.75f);

		// Draw Opaque
		{
			auto batch = scene.Batch(AlphaMode::Opaque);

			std::vector<Buffer *> instances;
			instances.reserve(batch.meshes.size());
			for (auto &mesh : batch.meshes)
			{
				instances.push_back(mesh->GetBuffer());
			}

			if (!instances.empty())
			{
				cmd_buffer.Bind(opaque_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, instances)
				                    .Bind(0, 1, scene.GetAssetManager().GetMeshletBuffer())
				                    .Bind(0, 2, scene.GetAssetManager().GetVertexBuffer())
				                    .Bind(0, 3, scene.GetAssetManager().GetMeshletVertexBuffer())
				                    .Bind(0, 4, scene.GetAssetManager().GetMeshletTriangleBuffer()));

				uint32_t instance_id = 0;
				for (auto &mesh : batch.meshes)
				{
					push_data.instance_id   = instance_id;
					push_data.meshlet_count = mesh->GetMesh()->GetMeshletsCount();
					cmd_buffer.PushConstants(VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_TASK_BIT_NV, &push_data, sizeof(push_data), 0);
					vkCmdDrawMeshTasksNV(cmd_buffer, (push_data.meshlet_count + 32 - 1) / 32, 0);
					instance_id++;
				}
			}
		}

		// Draw Alpha
		{
			auto batch = scene.Batch(AlphaMode::Masked | AlphaMode::Blend);

			std::vector<Buffer *> instances;
			instances.reserve(batch.meshes.size());
			for (auto &mesh : batch.meshes)
			{
				instances.push_back(mesh->GetBuffer());
			}

			if (!instances.empty())
			{
				cmd_buffer.Bind(alpha_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, instances)
				                    .Bind(0, 1, scene.GetAssetManager().GetMeshletBuffer())
				                    .Bind(0, 2, scene.GetAssetManager().GetVertexBuffer())
				                    .Bind(0, 3, scene.GetAssetManager().GetMeshletVertexBuffer())
				                    .Bind(0, 4, scene.GetAssetManager().GetMeshletTriangleBuffer())
				                    .Bind(0, 5, scene.GetAssetManager().GetMaterialBuffer())
				                    .Bind(0, 6, scene.GetAssetManager().GetTextureViews())
				                    .Bind(0, 7, device->AllocateSampler(SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_LINEAR})));

				uint32_t instance_id = 0;
				for (uint32_t i = 0; i < batch.order.size(); i++)
				{
					push_data.instance_id   = instance_id;
					push_data.meshlet_count = batch.meshes[i]->GetMesh()->GetMeshletsCount();
					cmd_buffer.PushConstants(VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_TASK_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT, &push_data, sizeof(push_data), 0);
					vkCmdDrawMeshTasksNV(cmd_buffer, (push_data.meshlet_count + 32 - 1) / 32, 0);
				}
			}
		}

		cmd_buffer.EndRenderPass();
		cmd_buffer.Transition(m_shadow_map.get(), TextureState{VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT}, TextureState{VK_IMAGE_USAGE_SAMPLED_BIT}, VkImageSubresourceRange{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1});
		cmd_buffer.End();
		device->Submit(cmd_buffer);
	}
}

void Light::UpdateAreaLight(Scene &scene, entt::entity entity, RHIDevice *device)
{
	if (!m_buffer || m_buffer->GetSize() != sizeof(ShaderInterop::AreaLight))
	{
		m_buffer = std::make_unique<Buffer>(device, BufferDesc(sizeof(ShaderInterop::AreaLight), 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU));
	}

	Entity e = Entity(scene, entity);

	auto &transform = e.GetComponent<cmpt::Transform>();

	ShaderInterop::AreaLight *data = static_cast<ShaderInterop::AreaLight *>(m_buffer->Map());

	data->corners[0] = transform.GetWorldTransform() * glm::vec4(-1.f, -1.f, 0.f, 1.f);
	data->corners[1] = transform.GetWorldTransform() * glm::vec4(1.f, -1.f, 0.f, 1.f);
	data->corners[2] = transform.GetWorldTransform() * glm::vec4(1.f, 1.f, 0.f, 1.f);
	data->corners[3] = transform.GetWorldTransform() * glm::vec4(-1.f, 1.f, 0.f, 1.f);

	m_buffer->Flush(m_buffer->GetSize());
	m_buffer->Unmap();
}

}        // namespace Ilum::cmpt