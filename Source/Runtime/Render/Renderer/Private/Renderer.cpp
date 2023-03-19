#include "Renderer.hpp"
#include "RenderData.hpp"

#include <Core/Path.hpp>
#include <Material/MaterialCompiler.hpp>
#include <Material/MaterialData.hpp>
#include <RHI/RHIContext.hpp>
#include <RenderGraph/RenderGraph.hpp>
#include <RenderGraph/RenderGraphBlackboard.hpp>
#include <Resource/Resource/Animation.hpp>
#include <Resource/Resource/Material.hpp>
#include <Resource/Resource/Mesh.hpp>
#include <Resource/Resource/SkinnedMesh.hpp>
#include <Resource/Resource/Texture2D.hpp>
#include <Resource/Resource/TextureCube.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Components/AllComponents.hpp>
#include <Scene/Node.hpp>
#include <Scene/Scene.hpp>
#include <ShaderCompiler/ShaderBuilder.hpp>
#include <ShaderCompiler/ShaderCompiler.hpp>

#include <cereal/types/vector.hpp>

namespace Ilum
{
struct Renderer::Impl
{
	RHIContext *rhi_context = nullptr;

	Scene *scene = nullptr;

	ResourceManager *resource_manager = nullptr;

	glm::vec2 viewport = {1.f, 1.f};

	RHITexture *present_texture = nullptr;

	std::unique_ptr<RenderGraph> render_graph     = nullptr;
	std::unique_ptr<RenderGraph> new_render_graph = nullptr;

	RenderGraphBlackboard black_board;

	std::unique_ptr<ShaderBuilder> shader_builder = nullptr;

	std::unique_ptr<RHIPipelineState> gpu_skinning_pipeline = nullptr;

	ShaderMeta gpu_skinning_shader_meta;

	float animation_time   = 0.f;
	bool  update_animation = false;

	Cmpt::Camera *main_camera = nullptr;
};

Renderer::Renderer(RHIContext *rhi_context, Scene *scene, ResourceManager *resource_manager)
{
	m_impl                        = new Impl;
	m_impl->rhi_context           = rhi_context;
	m_impl->scene                 = scene;
	m_impl->resource_manager      = resource_manager;
	m_impl->shader_builder        = std::make_unique<ShaderBuilder>(rhi_context);
	m_impl->gpu_skinning_pipeline = rhi_context->CreatePipelineState();

	// GPU Skinning Pipeline
	{
		RHIShader *shader = m_impl->shader_builder->RequireShader("./Source/Shaders/UpdateBoneMatrics.hlsl", "CSmain", RHIShaderStage::Compute);
		m_impl->gpu_skinning_pipeline->SetShader(RHIShaderStage::Compute, shader);
		m_impl->gpu_skinning_shader_meta = m_impl->shader_builder->RequireShaderMeta(shader);
	}

	// View
	{
		auto *view   = m_impl->black_board.Add<View>();
		view->buffer = m_impl->rhi_context->CreateBuffer<View::Info>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);
	}

	// GPU Scene
	{
		auto *gpu_scene            = m_impl->black_board.Add<GPUScene>();
		gpu_scene->opaque_tlas     = m_impl->rhi_context->CreateAcccelerationStructure();
		gpu_scene->non_opaque_tlas = m_impl->rhi_context->CreateAcccelerationStructure();

		gpu_scene->animation.update_info = m_impl->rhi_context->CreateBuffer<GPUScene::AnimationBuffer::UpdateInfo>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);
	}

	// Dummy Textures
	{
		auto *dummy_texture              = m_impl->black_board.Add<DummyTexture>();
		dummy_texture->white_opaque      = m_impl->rhi_context->CreateTexture2D(1, 1, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::Transfer | RHITextureUsage::ShaderResource, false);
		dummy_texture->black_opaque      = m_impl->rhi_context->CreateTexture2D(1, 1, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::Transfer | RHITextureUsage::ShaderResource, false);
		dummy_texture->white_transparent = m_impl->rhi_context->CreateTexture2D(1, 1, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::Transfer | RHITextureUsage::ShaderResource, false);
		dummy_texture->black_transparent = m_impl->rhi_context->CreateTexture2D(1, 1, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::Transfer | RHITextureUsage::ShaderResource, false);

		auto white_opaque_buffer      = m_impl->rhi_context->CreateBuffer<glm::vec4>(1, RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU);
		auto black_opaque_buffer      = m_impl->rhi_context->CreateBuffer<glm::vec4>(1, RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU);
		auto white_transparent_buffer = m_impl->rhi_context->CreateBuffer<glm::vec4>(1, RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU);
		auto black_transparent_buffer = m_impl->rhi_context->CreateBuffer<glm::vec4>(1, RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU);

		glm::vec4 white_opaque      = {1.f, 1.f, 1.f, 1.f};
		glm::vec4 black_opaque      = {0.f, 0.f, 0.f, 1.f};
		glm::vec4 white_transparent = {1.f, 1.f, 1.f, 0.f};
		glm::vec4 black_transparent = {0.f, 0.f, 0.f, 0.f};

		white_opaque_buffer->CopyToDevice(&white_opaque, sizeof(white_opaque));
		black_opaque_buffer->CopyToDevice(&black_opaque, sizeof(white_opaque));
		white_transparent_buffer->CopyToDevice(&white_transparent, sizeof(white_opaque));
		black_transparent_buffer->CopyToDevice(&black_transparent, sizeof(white_opaque));

		auto *cmd_buffer = m_impl->rhi_context->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->ResourceStateTransition({TextureStateTransition{dummy_texture->white_opaque.get(), RHIResourceState::Undefined, RHIResourceState::TransferDest, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}},
		                                     TextureStateTransition{dummy_texture->black_opaque.get(), RHIResourceState::Undefined, RHIResourceState::TransferDest, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}},
		                                     TextureStateTransition{dummy_texture->white_transparent.get(), RHIResourceState::Undefined, RHIResourceState::TransferDest, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}},
		                                     TextureStateTransition{dummy_texture->black_transparent.get(), RHIResourceState::Undefined, RHIResourceState::TransferDest, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
		                                    {});
		cmd_buffer->CopyBufferToTexture(white_opaque_buffer.get(), dummy_texture->white_opaque.get(), 0, 0, 1);
		cmd_buffer->CopyBufferToTexture(black_opaque_buffer.get(), dummy_texture->black_opaque.get(), 0, 0, 1);
		cmd_buffer->CopyBufferToTexture(white_transparent_buffer.get(), dummy_texture->white_transparent.get(), 0, 0, 1);
		cmd_buffer->CopyBufferToTexture(black_transparent_buffer.get(), dummy_texture->black_transparent.get(), 0, 0, 1);
		cmd_buffer->ResourceStateTransition({TextureStateTransition{dummy_texture->white_opaque.get(), RHIResourceState::TransferDest, RHIResourceState::ShaderResource, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}},
		                                     TextureStateTransition{dummy_texture->black_opaque.get(), RHIResourceState::TransferDest, RHIResourceState::ShaderResource, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}},
		                                     TextureStateTransition{dummy_texture->white_transparent.get(), RHIResourceState::TransferDest, RHIResourceState::ShaderResource, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}},
		                                     TextureStateTransition{dummy_texture->black_transparent.get(), RHIResourceState::TransferDest, RHIResourceState::ShaderResource, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
		                                    {});
		cmd_buffer->End();
		m_impl->rhi_context->Execute(cmd_buffer);
	}

	// LUT
	{
		auto *lut = m_impl->black_board.Add<LUT>();
		lut->ggx  = rhi_context->CreateTexture2D(512, 512, RHIFormat::R16G16_FLOAT, RHITextureUsage::ShaderResource | RHITextureUsage::UnorderedAccess, false);

		{
			std::unique_ptr<RHIPipelineState> ggx_preintegration = rhi_context->CreatePipelineState();

			RHIShader *shader = RequireShader("Source/Shaders/PreProcess/GGXBRDFLUT.hlsl", "CSmain", RHIShaderStage::Compute);
			ggx_preintegration->SetShader(RHIShaderStage::Compute, shader);
			ShaderMeta meta = RequireShaderMeta(shader);

			auto *descriptor = rhi_context->CreateDescriptor(meta);
			descriptor->BindTexture("Output", lut->ggx.get(), RHITextureDimension::Texture2D);

			auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Compute);
			cmd_buffer->Begin();
			cmd_buffer->ResourceStateTransition({
			                                        TextureStateTransition{lut->ggx.get(), RHIResourceState::Undefined, RHIResourceState::UnorderedAccess},
			                                    },
			                                    {});
			cmd_buffer->BindDescriptor(descriptor);
			cmd_buffer->BindPipelineState(ggx_preintegration.get());
			cmd_buffer->Dispatch(512, 512, 1, 8, 8, 1);
			cmd_buffer->ResourceStateTransition({
			                                        TextureStateTransition{lut->ggx.get(), RHIResourceState::UnorderedAccess, RHIResourceState::ShaderResource},
			                                    },
			                                    {});
			cmd_buffer->End();
			m_impl->rhi_context->Execute(cmd_buffer);
		}
	}
}

Renderer::~Renderer()
{
	delete m_impl;
}

void Renderer::Tick()
{
	m_impl->present_texture = nullptr;

	m_impl->scene->Update();

	UpdateGPUScene();

	if (m_impl->new_render_graph)
	{
		m_impl->rhi_context->WaitIdle();
		m_impl->render_graph = std::move(m_impl->new_render_graph);
		m_impl->new_render_graph.reset();
	}

	if (m_impl->render_graph)
	{
		m_impl->render_graph->Execute(m_impl->black_board);
	}

	m_impl->scene->Reset();
}

void Renderer::SetRenderGraph(std::unique_ptr<RenderGraph> &&render_graph)
{
	m_impl->new_render_graph = std::move(render_graph);
}

RenderGraph *Renderer::GetRenderGraph() const
{
	return m_impl->render_graph.get();
}

RHIContext *Renderer::GetRHIContext() const
{
	return m_impl->rhi_context;
}

ResourceManager *Renderer::GetResourceManager() const
{
	return m_impl->resource_manager;
}

RenderGraphBlackboard &Renderer::GetRenderGraphBlackboard()
{
	return m_impl->black_board;
}

void Renderer::SetViewport(float width, float height)
{
	m_impl->viewport = glm::vec2{width, height};
}

glm::vec2 Renderer::GetViewport() const
{
	return m_impl->viewport;
}

void Renderer::SetAnimationTime(float time)
{
	m_impl->animation_time   = time;
	m_impl->update_animation = true;
}

void Renderer::SetPresentTexture(RHITexture *present_texture)
{
	m_impl->present_texture = present_texture;
}

RHITexture *Renderer::GetPresentTexture() const
{
	return m_impl->present_texture ? m_impl->present_texture : m_impl->black_board.Get<DummyTexture>()->black_opaque.get();
}

float Renderer::GetMaxAnimationTime() const
{
	return static_cast<float>(m_impl->black_board.Get<GPUScene>()->animation.max_frame_count) / 30.f;
}

void Renderer::UpdateView(Cmpt::Camera *camera)
{
	m_impl->main_camera = camera;

	if (camera)
	{
		auto *view = m_impl->black_board.Get<View>();

		View::Info view_info = {};

		std::memcpy(view_info.frustum, camera->GetFrustumPlanes().data(), sizeof(glm::vec4) * 6);
		view_info.position                   = camera->GetNode()->GetComponent<Cmpt::Transform>()->GetWorldTransform()[3];
		view_info.projection_matrix          = camera->GetProjectionMatrix();
		view_info.view_matrix                = camera->GetViewMatrix();
		view_info.inv_projection_matrix      = camera->GetInvProjectionMatrix();
		view_info.inv_view_matrix            = camera->GetInvViewMatrix();
		view_info.view_projection_matrix     = camera->GetViewProjectionMatrix();
		view_info.inv_view_projection_matrix = camera->GetInvViewProjectionMatrix();
		view_info.viewport                   = m_impl->viewport;

		view->buffer->CopyToDevice(&view_info, sizeof(View::Info));
	}
}

Scene *Renderer::GetScene() const
{
	return m_impl->scene;
}

void Renderer::Reset()
{
	m_impl->rhi_context->WaitIdle();
}

RHIShader *Renderer::RequireShader(const std::string &filename, const std::string &entry_point, RHIShaderStage stage, std::vector<std::string> &&macros, std::vector<std::string> &&includes, bool cuda, bool force_recompile)
{
	return m_impl->shader_builder->RequireShader(filename, entry_point, stage, std::move(macros), std::move(includes), cuda, force_recompile);
}

ShaderMeta Renderer::RequireShaderMeta(RHIShader *shader) const
{
	return m_impl->shader_builder->RequireShaderMeta(shader);
}

void Renderer::UpdateLight()
{
	auto *gpu_scene = m_impl->black_board.Get<GPUScene>();

	auto point_lights       = m_impl->scene->GetComponents<Cmpt::PointLight>();
	auto spot_lights        = m_impl->scene->GetComponents<Cmpt::SpotLight>();
	auto directional_lights = m_impl->scene->GetComponents<Cmpt::DirectionalLight>();
	auto rectangle_lights   = m_impl->scene->GetComponents<Cmpt::RectLight>();
	auto environment_lights = m_impl->scene->GetComponents<Cmpt::EnvironmentLight>();

	if (!gpu_scene->light.light_info_buffer)
	{
		gpu_scene->light.light_info_buffer = m_impl->rhi_context->CreateBuffer<GPUScene::LightBuffer::Info>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);
	}

#define COPY_LIGHT_DATA(DATA, BUFFER)                                                                                                             \
	if (!DATA.empty())                                                                                                                            \
	{                                                                                                                                             \
		size_t light_size = DATA.empty() ? 0 : DATA.size() * DATA[0]->GetDataSize();                                                              \
		if (!gpu_scene->light.BUFFER || gpu_scene->light.BUFFER->GetDesc().size != light_size)                                                    \
		{                                                                                                                                         \
			gpu_scene->light.BUFFER = m_impl->rhi_context->CreateBuffer(light_size, RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU); \
		}                                                                                                                                         \
		auto    *light_buffer = gpu_scene->light.BUFFER->Map();                                                                                   \
		size_t   offset       = 0;                                                                                                                \
		uint32_t shadow_id    = 0;                                                                                                                \
		for (auto &light : DATA)                                                                                                                  \
		{                                                                                                                                         \
			light->SetShadowID(shadow_id);                                                                                                        \
			std::memcpy((uint8_t *) light_buffer + offset, light->GetData(m_impl->main_camera), light->GetDataSize());                            \
			offset += light->GetDataSize();                                                                                                       \
		}                                                                                                                                         \
		gpu_scene->light.BUFFER->Unmap();                                                                                                         \
	}                                                                                                                                             \
	else                                                                                                                                          \
	{                                                                                                                                             \
		if (!gpu_scene->light.BUFFER)                                                                                                             \
		{                                                                                                                                         \
			gpu_scene->light.BUFFER = m_impl->rhi_context->CreateBuffer(1, RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);          \
		}                                                                                                                                         \
	}

	COPY_LIGHT_DATA(point_lights, point_light_buffer);
	COPY_LIGHT_DATA(spot_lights, spot_light_buffer);
	COPY_LIGHT_DATA(directional_lights, directional_light_buffer);
	COPY_LIGHT_DATA(rectangle_lights, rect_light_buffer);

	gpu_scene->light.info.point_light_count       = static_cast<uint32_t>(point_lights.size());
	gpu_scene->light.info.spot_light_count        = static_cast<uint32_t>(spot_lights.size());
	gpu_scene->light.info.directional_light_count = static_cast<uint32_t>(directional_lights.size());
	gpu_scene->light.info.rect_light_count        = static_cast<uint32_t>(rectangle_lights.size());
	gpu_scene->light.light_info_buffer->CopyToDevice(&gpu_scene->light.info, sizeof(gpu_scene->light.info));

	// Copy environment light data
	if (environment_lights.empty())
	{
		gpu_scene->texture.texture_cube = nullptr;
	}
	else
	{
		auto resource = m_impl->resource_manager->Get<ResourceType::TextureCube>(static_cast<const char *>(environment_lights.back()->GetData()));
		if (resource)
		{
			gpu_scene->texture.texture_cube = resource->GetTexture();
		}
		else
		{
			gpu_scene->texture.texture_cube = nullptr;
		}
	}
}

void Renderer::UpdateAnimation()
{
	auto *gpu_scene = m_impl->black_board.Get<GPUScene>();

	// Update animation
	if (gpu_scene->animation.max_bone_count > 0 && m_impl->update_animation)
	{
		{
			GPUScene::AnimationBuffer::UpdateInfo update_info = {};

			update_info.count = static_cast<uint32_t>(gpu_scene->animation.bone_matrics.size());
			update_info.time  = m_impl->animation_time;
			gpu_scene->animation.update_info->CopyToDevice(&update_info, sizeof(update_info));
		}

		std::vector<BufferStateTransition> begin_transitions(gpu_scene->animation.bone_matrics.size());
		std::vector<BufferStateTransition> end_transitions(gpu_scene->animation.bone_matrics.size());

		for (size_t i = 0; i < gpu_scene->animation.bone_matrics.size(); i++)
		{
			begin_transitions[i] = BufferStateTransition{gpu_scene->animation.bone_matrics[i], RHIResourceState::ShaderResource, RHIResourceState::UnorderedAccess};
			end_transitions[i]   = BufferStateTransition{gpu_scene->animation.bone_matrics[i], RHIResourceState::UnorderedAccess, RHIResourceState::ShaderResource};
		}

		auto *descriptor = m_impl->rhi_context->CreateDescriptor(m_impl->gpu_skinning_shader_meta);
		descriptor->BindBuffer("UpdateInfo", gpu_scene->animation.update_info.get());
		auto *cmd_buffer = m_impl->rhi_context->CreateCommand(RHIQueueFamily::Compute);
		cmd_buffer->Begin();
		cmd_buffer->ResourceStateTransition({}, begin_transitions);
		for (size_t i = 0; i < gpu_scene->animation.bone_matrics.size(); i++)
		{
			descriptor->BindBuffer("BoneMatrics", gpu_scene->animation.bone_matrics[i])
			    .BindTexture("SkinnedMatrics", gpu_scene->animation.skinned_matrics[i], RHITextureDimension::Texture2D);
			cmd_buffer->BindDescriptor(descriptor);
			cmd_buffer->BindPipelineState(m_impl->gpu_skinning_pipeline.get());
			cmd_buffer->Dispatch(gpu_scene->animation.max_bone_count, 1, 1, 8, 1, 1);
		}
		cmd_buffer->ResourceStateTransition({}, end_transitions);
		cmd_buffer->End();
		m_impl->rhi_context->Submit({cmd_buffer});
		m_impl->update_animation = false;
	}

	// Update resource
	if (m_impl->resource_manager->Update<ResourceType::Animation>())
	{
		m_impl->scene->Update(true);

		gpu_scene->animation.bone_matrics.clear();
		gpu_scene->animation.skinned_matrics.clear();

		auto resources = m_impl->resource_manager->GetResources<ResourceType::Animation>();

		gpu_scene->animation.max_bone_count  = 0;
		gpu_scene->animation.max_frame_count = 0;
		for (auto &resource : resources)
		{
			auto *animation = m_impl->resource_manager->Get<ResourceType::Animation>(resource);
			gpu_scene->animation.skinned_matrics.push_back(animation->GetSkinnedMatrics());
			gpu_scene->animation.bone_matrics.push_back(animation->GetBoneMatrics());
			gpu_scene->animation.max_bone_count  = glm::max(gpu_scene->animation.max_bone_count, animation->GetBoneCount());
			gpu_scene->animation.max_frame_count = glm::max(gpu_scene->animation.max_frame_count, animation->GetFrameCount());
		}
	}
}

void Renderer::UpdateMesh()
{
	auto *gpu_scene = m_impl->black_board.Get<GPUScene>();

	auto meshes = m_impl->scene->GetComponents<Cmpt::MeshRenderer>();

	TLASDesc opaque_tlas_desc     = {fmt::format("{} - Opaque", m_impl->scene->GetName())};
	TLASDesc non_opaque_tlas_desc = {fmt::format("{} - Non Opaque", m_impl->scene->GetName())};

	std::vector<GPUScene::Instance> opaque_instances;
	std::vector<GPUScene::Instance> non_opaque_instances;

	gpu_scene->opaque_mesh.max_meshlet_count     = 0;
	gpu_scene->non_opaque_mesh.max_meshlet_count = 0;

	// Update mesh instances
	{
		for (auto &mesh : meshes)
		{
			auto &submeshes = mesh->GetSubmeshes();
			auto &materials = mesh->GetMaterials();
			for (uint32_t i = 0; i < submeshes.size(); i++)
			{
				auto &submesh = submeshes[i];

				auto *resource = m_impl->resource_manager->Get<ResourceType::Mesh>(submesh);

				if (resource)
				{
					GPUScene::Instance instance = {};
					instance.transform          = mesh->GetNode()->GetComponent<Cmpt::Transform>()->GetWorldTransform();
					instance.mesh_id            = static_cast<uint32_t>(m_impl->resource_manager->Index<ResourceType::Mesh>(submesh));
					instance.material_id        = 0;

					if (i < materials.size())
					{
						instance.material_id = static_cast<uint32_t>(m_impl->resource_manager->Index<ResourceType::Material>(materials[i])) + 1;
						auto *material       = m_impl->resource_manager->Get<ResourceType::Material>(materials[i]);
						if (material->GetMaterialData().blend_mode == BlendMode::Opaque)
						{
							opaque_instances.push_back(instance);
							opaque_tlas_desc.instances.push_back(TLASDesc::InstanceInfo{instance.transform, instance.material_id, resource->GetBLAS()});
							gpu_scene->opaque_mesh.max_meshlet_count = glm::max(gpu_scene->opaque_mesh.max_meshlet_count, static_cast<uint32_t>(resource->GetMeshletCount()));
						}
						else
						{
							non_opaque_instances.push_back(instance);
							non_opaque_tlas_desc.instances.push_back(TLASDesc::InstanceInfo{instance.transform, instance.material_id, resource->GetBLAS()});
							gpu_scene->non_opaque_mesh.max_meshlet_count = glm::max(gpu_scene->non_opaque_mesh.max_meshlet_count, static_cast<uint32_t>(resource->GetMeshletCount()));
						}
					}
					else
					{
						opaque_instances.push_back(instance);
						opaque_tlas_desc.instances.push_back(TLASDesc::InstanceInfo{instance.transform, instance.material_id, resource->GetBLAS()});
						gpu_scene->opaque_mesh.max_meshlet_count = glm::max(gpu_scene->opaque_mesh.max_meshlet_count, static_cast<uint32_t>(resource->GetMeshletCount()));
					}
				}
			}
		}

		gpu_scene->opaque_mesh.instance_count     = static_cast<uint32_t>(opaque_instances.size());
		gpu_scene->non_opaque_mesh.instance_count = static_cast<uint32_t>(non_opaque_instances.size());
	}

	// Copy to device
	{
		if (!opaque_instances.empty())
		{
			if (!gpu_scene->opaque_mesh.instances ||
			    gpu_scene->opaque_mesh.instances->GetDesc().size < opaque_instances.size() * sizeof(GPUScene::Instance))
			{
				gpu_scene->opaque_mesh.instances = m_impl->rhi_context->CreateBuffer<GPUScene::Instance>(opaque_instances.size(), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
			}
			gpu_scene->opaque_mesh.instances->CopyToDevice(opaque_instances.data(), opaque_instances.size() * sizeof(GPUScene::Instance));
		}

		if (!non_opaque_instances.empty())
		{
			if (!gpu_scene->non_opaque_mesh.instances ||
			    gpu_scene->non_opaque_mesh.instances->GetDesc().size < non_opaque_instances.size() * sizeof(GPUScene::Instance))
			{
				gpu_scene->non_opaque_mesh.instances = m_impl->rhi_context->CreateBuffer<GPUScene::Instance>(non_opaque_instances.size(), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
			}
			gpu_scene->non_opaque_mesh.instances->CopyToDevice(non_opaque_instances.data(), non_opaque_instances.size() * sizeof(GPUScene::Instance));
		}
	}

	// Update TLAS
	{
		if (!opaque_tlas_desc.instances.empty())
		{
			auto *cmd_buffer = m_impl->rhi_context->CreateCommand(RHIQueueFamily::Compute);
			cmd_buffer->Begin();
			gpu_scene->opaque_tlas->Update(cmd_buffer, opaque_tlas_desc);
			cmd_buffer->End();
			m_impl->rhi_context->Submit({cmd_buffer});
		}

		if (!non_opaque_tlas_desc.instances.empty())
		{
			auto *cmd_buffer = m_impl->rhi_context->CreateCommand(RHIQueueFamily::Compute);
			cmd_buffer->Begin();
			gpu_scene->non_opaque_tlas->Update(cmd_buffer, non_opaque_tlas_desc);
			cmd_buffer->End();
			m_impl->rhi_context->Submit({cmd_buffer});
		}
	}

	// Update resource
	if (m_impl->resource_manager->Update<ResourceType::Mesh>())
	{
		m_impl->scene->Update(true);

		gpu_scene->mesh_buffer.vertex_buffers.clear();
		gpu_scene->mesh_buffer.index_buffers.clear();
		gpu_scene->mesh_buffer.meshlet_buffers.clear();
		gpu_scene->mesh_buffer.meshlet_data_buffers.clear();
		auto resources = m_impl->resource_manager->GetResources<ResourceType::Mesh>();
		for (auto &resource : resources)
		{
			auto *mesh = m_impl->resource_manager->Get<ResourceType::Mesh>(resource);
			gpu_scene->mesh_buffer.vertex_buffers.push_back(mesh->GetVertexBuffer());
			gpu_scene->mesh_buffer.index_buffers.push_back(mesh->GetIndexBuffer());
			gpu_scene->mesh_buffer.meshlet_buffers.push_back(mesh->GetMeshletBuffer());
			gpu_scene->mesh_buffer.meshlet_data_buffers.push_back(mesh->GetMeshletDataBuffer());
		}
	}
}

void Renderer::UpdateSkinnedMesh()
{
	auto *gpu_scene      = m_impl->black_board.Get<GPUScene>();
	auto  skinned_meshes = m_impl->scene->GetComponents<Cmpt::SkinnedMeshRenderer>();

	std::vector<GPUScene::Instance> opaque_instances;
	std::vector<GPUScene::Instance> non_opaque_instances;

	gpu_scene->opaque_skinned_mesh.max_meshlet_count     = 0;
	gpu_scene->non_opaque_skinned_mesh.max_meshlet_count = 0;

	// Update skinned mesh instances
	{
		for (auto &skinned_mesh : skinned_meshes)
		{
			auto &submeshes  = skinned_mesh->GetSubmeshes();
			auto &animations = skinned_mesh->GetAnimations();
			auto &materials  = skinned_mesh->GetMaterials();
			for (uint32_t i = 0; i < submeshes.size(); i++)
			{
				auto *resource = m_impl->resource_manager->Get<ResourceType::SkinnedMesh>(submeshes[i]);

				if (resource)
				{
					GPUScene::Instance instance = {};
					instance.transform          = skinned_mesh->GetNode()->GetComponent<Cmpt::Transform>()->GetWorldTransform();
					instance.mesh_id            = static_cast<uint32_t>(m_impl->resource_manager->Index<ResourceType::SkinnedMesh>(submeshes[i]));
					instance.material_id        = 0;

					if (i < animations.size())
					{
						instance.animation_id = static_cast<uint32_t>(m_impl->resource_manager->Index<ResourceType::Animation>(animations[i]));
					}

					if (i < materials.size())
					{
						instance.material_id = static_cast<uint32_t>(m_impl->resource_manager->Index<ResourceType::Material>(materials[i])) + 1;
						auto *material       = m_impl->resource_manager->Get<ResourceType::Material>(materials[i]);
						if (material->GetMaterialData().blend_mode == BlendMode::Opaque)
						{
							opaque_instances.push_back(instance);
							gpu_scene->opaque_mesh.max_meshlet_count = glm::max(gpu_scene->opaque_mesh.max_meshlet_count, static_cast<uint32_t>(resource->GetMeshletCount()));
						}
						else
						{
							non_opaque_instances.push_back(instance);
							gpu_scene->non_opaque_mesh.max_meshlet_count = glm::max(gpu_scene->non_opaque_mesh.max_meshlet_count, static_cast<uint32_t>(resource->GetMeshletCount()));
						}
					}
					else
					{
						opaque_instances.push_back(instance);
						gpu_scene->opaque_mesh.max_meshlet_count = glm::max(gpu_scene->opaque_mesh.max_meshlet_count, static_cast<uint32_t>(resource->GetMeshletCount()));
					}
				}
			}
		}

		gpu_scene->opaque_skinned_mesh.instance_count     = static_cast<uint32_t>(opaque_instances.size());
		gpu_scene->non_opaque_skinned_mesh.instance_count = static_cast<uint32_t>(non_opaque_instances.size());
	}

	// Copy to device
	{
		if (!opaque_instances.empty())
		{
			if (!gpu_scene->opaque_skinned_mesh.instances ||
			    gpu_scene->opaque_skinned_mesh.instances->GetDesc().size < opaque_instances.size() * sizeof(GPUScene::Instance))
			{
				gpu_scene->opaque_skinned_mesh.instances = m_impl->rhi_context->CreateBuffer<GPUScene::Instance>(opaque_instances.size(), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
			}
			gpu_scene->opaque_skinned_mesh.instances->CopyToDevice(opaque_instances.data(), opaque_instances.size() * sizeof(GPUScene::Instance));
		}

		if (!non_opaque_instances.empty())
		{
			if (!gpu_scene->non_opaque_skinned_mesh.instances ||
			    gpu_scene->non_opaque_skinned_mesh.instances->GetDesc().size < non_opaque_instances.size() * sizeof(GPUScene::Instance))
			{
				gpu_scene->non_opaque_skinned_mesh.instances = m_impl->rhi_context->CreateBuffer<GPUScene::Instance>(non_opaque_instances.size(), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
			}
			gpu_scene->non_opaque_skinned_mesh.instances->CopyToDevice(non_opaque_instances.data(), non_opaque_instances.size() * sizeof(GPUScene::Instance));
		}
	}

	// Update resource
	if (m_impl->resource_manager->Update<ResourceType::SkinnedMesh>())
	{
		m_impl->scene->Update(true);

		gpu_scene->skinned_mesh_buffer.vertex_buffers.clear();
		gpu_scene->skinned_mesh_buffer.index_buffers.clear();
		gpu_scene->skinned_mesh_buffer.meshlet_buffers.clear();
		gpu_scene->skinned_mesh_buffer.meshlet_data_buffers.clear();
		auto resources = m_impl->resource_manager->GetResources<ResourceType::SkinnedMesh>();
		for (auto &resource : resources)
		{
			auto *skinned_mesh = m_impl->resource_manager->Get<ResourceType::SkinnedMesh>(resource);
			gpu_scene->skinned_mesh_buffer.vertex_buffers.push_back(skinned_mesh->GetVertexBuffer());
			gpu_scene->skinned_mesh_buffer.index_buffers.push_back(skinned_mesh->GetIndexBuffer());
			gpu_scene->skinned_mesh_buffer.meshlet_buffers.push_back(skinned_mesh->GetMeshletBuffer());
			gpu_scene->skinned_mesh_buffer.meshlet_data_buffers.push_back(skinned_mesh->GetMeshletDataBuffer());
		}
	}
}

void Renderer::UpdateMaterial()
{
	auto *gpu_scene = m_impl->black_board.Get<GPUScene>();

	// Update Sampler
	if (m_impl->rhi_context->GetSamplerCount() != gpu_scene->samplers.size())
	{
		m_impl->scene->Update(true);

		gpu_scene->samplers = m_impl->rhi_context->GetSamplers();
	}

	// Update 2D Textures
	if (m_impl->resource_manager->Update<ResourceType::Texture2D>())
	{
		m_impl->scene->Update(true);
	}

	// Update Material & Texture
	if (m_impl->resource_manager->Update<ResourceType::Material>() ||
	    m_impl->resource_manager->Update<ResourceType::Texture2D>())
	{
		m_impl->scene->Update(true);

		// Flush Texture
		if (m_impl->resource_manager->Update<ResourceType::Material>())
		{
			auto resources = m_impl->resource_manager->GetResources<ResourceType::Material>();
			for (auto &resource : resources)
			{
				auto *material = m_impl->resource_manager->Get<ResourceType::Material>(resource);
				if (!material->IsValid())
				{
					material->Compile(
					    m_impl->rhi_context,
					    m_impl->resource_manager,
					    m_impl->black_board.Get<DummyTexture>()->black_opaque.get());
				}
				for (auto &[texture, texture_name] : material->GetCompilationContext().textures)
				{
					m_impl->resource_manager->Index<ResourceType::Texture2D>(texture_name);
				}
			}
		}

		// Update Texture
		if (m_impl->resource_manager->Update<ResourceType::Texture2D>())
		{
			gpu_scene->texture.texture_2d.clear();
			auto resources = m_impl->resource_manager->GetResources<ResourceType::Texture2D>();
			for (auto &resource : resources)
			{
				auto *texture2d = m_impl->resource_manager->Get<ResourceType::Texture2D>(resource);
				gpu_scene->texture.texture_2d.push_back(texture2d->GetTexture());
			}
		}

		// Update Material
		if (m_impl->resource_manager->Update<ResourceType::Material>())
		{
			gpu_scene->material.data.clear();
			auto resources = m_impl->resource_manager->GetResources<ResourceType::Material>();

			std::vector<uint32_t> material_data;
			std::vector<uint32_t> material_offset = {0};

			for (auto &resource : resources)
			{
				auto *material = m_impl->resource_manager->Get<ResourceType::Material>(resource);

				material->Update(
				    m_impl->rhi_context,
				    m_impl->resource_manager,
				    m_impl->black_board.Get<DummyTexture>()->black_opaque.get());

				const auto &data = material->GetMaterialData();

				material_offset.push_back(static_cast<uint32_t>(material_data.size() * sizeof(uint32_t)));
				material_data.insert(material_data.end(), data.textures.begin(), data.textures.end());
				material_data.insert(material_data.end(), data.samplers.begin(), data.samplers.end());

				gpu_scene->material.data.push_back(&data);
			}

			if (!material_data.empty())
			{
				if (!gpu_scene->material.material_buffer ||
				    material_data.size() * sizeof(uint32_t) != gpu_scene->material.material_buffer->GetDesc().size)
				{
					gpu_scene->material.material_buffer = m_impl->rhi_context->CreateBuffer<uint32_t>(material_data.size(), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
				}
				gpu_scene->material.material_buffer->CopyToDevice(material_data.data(), material_data.size() * sizeof(uint32_t));
			}
			else if (!gpu_scene->material.material_buffer)
			{
				gpu_scene->material.material_buffer = m_impl->rhi_context->CreateBuffer<uint32_t>(1, RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
			}

			if (!material_offset.empty())
			{
				if (!gpu_scene->material.material_offset ||
				    material_offset.size() * sizeof(uint32_t) != gpu_scene->material.material_offset->GetDesc().size)
				{
					gpu_scene->material.material_offset = m_impl->rhi_context->CreateBuffer<uint32_t>(material_offset.size(), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
				}
				gpu_scene->material.material_offset->CopyToDevice(material_offset.data(), material_offset.size() * sizeof(uint32_t));
			}
			else if (!gpu_scene->material.material_offset)
			{
				gpu_scene->material.material_offset = m_impl->rhi_context->CreateBuffer<uint32_t>(1, RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
			}

			for (auto &resource : resources)
			{
				auto *material = m_impl->resource_manager->Get<ResourceType::Material>(resource);

				GPUScene *gpu_scene = m_impl->black_board.Get<GPUScene>();

				material->PostUpdate(
				    m_impl->rhi_context,
				    static_cast<uint32_t>(m_impl->resource_manager->Index<ResourceType::Material>(resource)),
				    gpu_scene->texture.texture_2d,
				    gpu_scene->samplers,
				    gpu_scene->material.material_buffer.get(),
				    gpu_scene->material.material_offset.get());
			}
		}
	}
}

void Renderer::UpdateGPUScene()
{
	auto *gpu_scene = m_impl->black_board.Get<GPUScene>();

	TLASDesc tlas_desc = {};
	tlas_desc.name     = m_impl->scene->GetName();

	auto meshes         = m_impl->scene->GetComponents<Cmpt::MeshRenderer>();
	auto skinned_meshes = m_impl->scene->GetComponents<Cmpt::SkinnedMeshRenderer>();

	UpdateLight();
	UpdateAnimation();
	UpdateMesh();
	UpdateSkinnedMesh();
	UpdateMaterial();

	// Update Mesh
	/*{
	    std::vector<GPUScene::Instance> instances;

	    gpu_scene->mesh_buffer.max_meshlet_count = 0;

	    for (auto &mesh : meshes)
	    {
	        auto &submeshes = mesh->GetSubmeshes();
	        auto &materials = mesh->GetMaterials();
	        for (uint32_t i = 0; i < submeshes.size(); i++)
	        {
	            auto &submesh = submeshes[i];

	            auto *resource = m_impl->resource_manager->Get<ResourceType::Mesh>(submesh);

	            if (resource)
	            {
	                GPUScene::Instance instance = {};
	                instance.transform          = mesh->GetNode()->GetComponent<Cmpt::Transform>()->GetWorldTransform();
	                instance.mesh_id            = static_cast<uint32_t>(m_impl->resource_manager->Index<ResourceType::Mesh>(submesh));
	                instance.material_id        = 0;

	                if (i < materials.size())
	                {
	                    instance.material_id = static_cast<uint32_t>(m_impl->resource_manager->Index<ResourceType::Material>(materials[i])) + 1;
	                }

	                instances.push_back(instance);

	                tlas_desc.instances.push_back(TLASDesc::InstanceInfo{instance.transform, instance.material_id, resource->GetBLAS()});
	                gpu_scene->mesh_buffer.max_meshlet_count = glm::max(gpu_scene->mesh_buffer.max_meshlet_count, static_cast<uint32_t>(resource->GetMeshletCount()));
	            }
	        }
	    }

	    gpu_scene->mesh_buffer.instance_count = static_cast<uint32_t>(instances.size());

	    if (!instances.empty())
	    {
	        if (!gpu_scene->mesh_buffer.instances ||
	            gpu_scene->mesh_buffer.instances->GetDesc().size < instances.size() * sizeof(GPUScene::Instance))
	        {
	            gpu_scene->mesh_buffer.instances = m_impl->rhi_context->CreateBuffer<GPUScene::Instance>(instances.size(), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
	        }

	        gpu_scene->mesh_buffer.instances->CopyToDevice(instances.data(), instances.size() * sizeof(GPUScene::Instance));
	    }
	}*/

	// Update Skinned Mesh
	{

	}

	// Update TLAS
	//{
	//	if (!tlas_desc.instances.empty())
	//	{
	//		auto *cmd_buffer = m_impl->rhi_context->CreateCommand(RHIQueueFamily::Compute);
	//		cmd_buffer->Begin();
	//		gpu_scene->TLAS->Update(cmd_buffer, tlas_desc);
	//		cmd_buffer->End();
	//		m_impl->rhi_context->Submit({cmd_buffer});
	//	}
	//}

	// Update Mesh Buffer
	// if (m_impl->resource_manager->Update<ResourceType::Mesh>())
	//{
	//	m_impl->scene->Update(true);

	//	gpu_scene->mesh_buffer.vertex_buffers.clear();
	//	gpu_scene->mesh_buffer.index_buffers.clear();
	//	gpu_scene->mesh_buffer.meshlet_buffers.clear();
	//	gpu_scene->mesh_buffer.meshlet_data_buffers.clear();
	//	auto resources = m_impl->resource_manager->GetResources<ResourceType::Mesh>();
	//	for (auto &resource : resources)
	//	{
	//		auto *mesh = m_impl->resource_manager->Get<ResourceType::Mesh>(resource);
	//		gpu_scene->mesh_buffer.vertex_buffers.push_back(mesh->GetVertexBuffer());
	//		gpu_scene->mesh_buffer.index_buffers.push_back(mesh->GetIndexBuffer());
	//		gpu_scene->mesh_buffer.meshlet_buffers.push_back(mesh->GetMeshletBuffer());
	//		gpu_scene->mesh_buffer.meshlet_data_buffers.push_back(mesh->GetMeshletDataBuffer());
	//	}
	//}

	// Update Skinned Mesh Buffer
	//{
	//	if (m_impl->resource_manager->Update<ResourceType::SkinnedMesh>())
	//	{
	//		m_impl->scene->Update(true);

	//		gpu_scene->skinned_mesh_buffer.vertex_buffers.clear();
	//		gpu_scene->skinned_mesh_buffer.index_buffers.clear();
	//		gpu_scene->skinned_mesh_buffer.meshlet_buffers.clear();
	//		gpu_scene->skinned_mesh_buffer.meshlet_data_buffers.clear();
	//		auto resources = m_impl->resource_manager->GetResources<ResourceType::SkinnedMesh>();
	//		for (auto &resource : resources)
	//		{
	//			auto *skinned_mesh = m_impl->resource_manager->Get<ResourceType::SkinnedMesh>(resource);
	//			gpu_scene->skinned_mesh_buffer.vertex_buffers.push_back(skinned_mesh->GetVertexBuffer());
	//			gpu_scene->skinned_mesh_buffer.index_buffers.push_back(skinned_mesh->GetIndexBuffer());
	//			gpu_scene->skinned_mesh_buffer.meshlet_buffers.push_back(skinned_mesh->GetMeshletBuffer());
	//			gpu_scene->skinned_mesh_buffer.meshlet_data_buffers.push_back(skinned_mesh->GetMeshletDataBuffer());
	//		}
	//	}
	//}

	//// Update Animation Buffer
	////{
	////	if (m_impl->resource_manager->Update<ResourceType::Animation>())
	////	{
	////		m_impl->scene->Update(true);

	////		gpu_scene->animation_buffer.bone_matrics.clear();
	////		gpu_scene->animation_buffer.skinned_matrics.clear();

	////		auto resources = m_impl->resource_manager->GetResources<ResourceType::Animation>();

	////		gpu_scene->animation_buffer.max_bone_count  = 0;
	////		gpu_scene->animation_buffer.max_frame_count = 0;
	////		for (auto &resource : resources)
	////		{
	////			auto *animation = m_impl->resource_manager->Get<ResourceType::Animation>(resource);
	////			gpu_scene->animation_buffer.skinned_matrics.push_back(animation->GetSkinnedMatrics());
	////			gpu_scene->animation_buffer.bone_matrics.push_back(animation->GetBoneMatrics());
	////			gpu_scene->animation_buffer.max_bone_count  = glm::max(gpu_scene->animation_buffer.max_bone_count, animation->GetBoneCount());
	////			gpu_scene->animation_buffer.max_frame_count = glm::max(gpu_scene->animation_buffer.max_frame_count, animation->GetFrameCount());
	////		}
	////	}
	////}

	//// Update Sampler
	//if (m_impl->rhi_context->GetSamplerCount() != gpu_scene->samplers.size())
	//{
	//	m_impl->scene->Update(true);

	//	gpu_scene->samplers = m_impl->rhi_context->GetSamplers();
	//}

	//// Update 2D Textures
	//if (m_impl->resource_manager->Update<ResourceType::Texture2D>())
	//{
	//	m_impl->scene->Update(true);
	//}

	//// Update Material & Texture
	//if (m_impl->resource_manager->Update<ResourceType::Material>() ||
	//    m_impl->resource_manager->Update<ResourceType::Texture2D>())
	//{
	//	m_impl->scene->Update(true);

	//	// Flush Texture
	//	if (m_impl->resource_manager->Update<ResourceType::Material>())
	//	{
	//		auto resources = m_impl->resource_manager->GetResources<ResourceType::Material>();
	//		for (auto &resource : resources)
	//		{
	//			auto *material = m_impl->resource_manager->Get<ResourceType::Material>(resource);
	//			if (!material->IsValid())
	//			{
	//				material->Compile(
	//				    m_impl->rhi_context,
	//				    m_impl->resource_manager,
	//				    m_impl->black_board.Get<DummyTexture>()->black_opaque.get());
	//			}
	//			for (auto &[texture, texture_name] : material->GetCompilationContext().textures)
	//			{
	//				m_impl->resource_manager->Index<ResourceType::Texture2D>(texture_name);
	//			}
	//		}
	//	}

	//	// Update Texture
	//	if (m_impl->resource_manager->Update<ResourceType::Texture2D>())
	//	{
	//		gpu_scene->textures.texture_2d.clear();
	//		auto resources = m_impl->resource_manager->GetResources<ResourceType::Texture2D>();
	//		for (auto &resource : resources)
	//		{
	//			auto *texture2d = m_impl->resource_manager->Get<ResourceType::Texture2D>(resource);
	//			gpu_scene->textures.texture_2d.push_back(texture2d->GetTexture());
	//		}
	//	}

	//	// Update Material
	//	if (m_impl->resource_manager->Update<ResourceType::Material>())
	//	{
	//		gpu_scene->material.data.clear();
	//		auto resources = m_impl->resource_manager->GetResources<ResourceType::Material>();

	//		std::vector<uint32_t> material_data;
	//		std::vector<uint32_t> material_offset = {0};

	//		for (auto &resource : resources)
	//		{
	//			auto *material = m_impl->resource_manager->Get<ResourceType::Material>(resource);

	//			material->Update(
	//			    m_impl->rhi_context,
	//			    m_impl->resource_manager,
	//			    m_impl->black_board.Get<DummyTexture>()->black_opaque.get());

	//			const auto &data = material->GetMaterialData();

	//			material_offset.push_back(static_cast<uint32_t>(material_data.size() * sizeof(uint32_t)));
	//			material_data.insert(material_data.end(), data.textures.begin(), data.textures.end());
	//			material_data.insert(material_data.end(), data.samplers.begin(), data.samplers.end());

	//			gpu_scene->material.data.push_back(&data);
	//		}

	//		if (!material_data.empty())
	//		{
	//			if (!gpu_scene->material.material_buffer ||
	//			    material_data.size() * sizeof(uint32_t) != gpu_scene->material.material_buffer->GetDesc().size)
	//			{
	//				gpu_scene->material.material_buffer = m_impl->rhi_context->CreateBuffer<uint32_t>(material_data.size(), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
	//			}
	//			gpu_scene->material.material_buffer->CopyToDevice(material_data.data(), material_data.size() * sizeof(uint32_t));
	//		}
	//		else if (!gpu_scene->material.material_buffer)
	//		{
	//			gpu_scene->material.material_buffer = m_impl->rhi_context->CreateBuffer<uint32_t>(1, RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
	//		}

	//		if (!material_offset.empty())
	//		{
	//			if (!gpu_scene->material.material_offset ||
	//			    material_offset.size() * sizeof(uint32_t) != gpu_scene->material.material_offset->GetDesc().size)
	//			{
	//				gpu_scene->material.material_offset = m_impl->rhi_context->CreateBuffer<uint32_t>(material_offset.size(), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
	//			}
	//			gpu_scene->material.material_offset->CopyToDevice(material_offset.data(), material_offset.size() * sizeof(uint32_t));
	//		}
	//		else if (!gpu_scene->material.material_offset)
	//		{
	//			gpu_scene->material.material_offset = m_impl->rhi_context->CreateBuffer<uint32_t>(1, RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
	//		}

	//		for (auto &resource : resources)
	//		{
	//			auto *material = m_impl->resource_manager->Get<ResourceType::Material>(resource);

	//			GPUScene *gpu_scene = m_impl->black_board.Get<GPUScene>();

	//			material->PostUpdate(
	//			    m_impl->rhi_context,
	//			    static_cast<uint32_t>(m_impl->resource_manager->Index<ResourceType::Material>(resource)),
	//			    gpu_scene->textures.texture_2d,
	//			    gpu_scene->samplers,
	//			    gpu_scene->material.material_buffer.get(),
	//			    gpu_scene->material.material_offset.get());
	//		}
	//	}
	//}
}
}        // namespace Ilum