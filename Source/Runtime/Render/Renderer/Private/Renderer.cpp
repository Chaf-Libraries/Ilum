#include "Renderer.hpp"
#include "RenderData.hpp"

#include <Components/AllComponents.hpp>
#include <Core/Path.hpp>
#include <RHI/RHIContext.hpp>
#include <RenderGraph/RenderGraph.hpp>
#include <Resource/Resource/Mesh.hpp>
#include <Resource/Resource/SkinnedMesh.hpp>
#include <Resource/Resource/Texture.hpp>
#include <Resource/ResourceManager.hpp>
#include <SceneGraph/Node.hpp>
#include <SceneGraph/Scene.hpp>
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

	glm::vec2 viewport = {};

	RHITexture *present_texture = nullptr;

	std::unique_ptr<RenderGraph> render_graph = nullptr;

	RenderGraphBlackboard black_board;

	std::unique_ptr<ShaderBuilder> shader_builder = nullptr;
};

Renderer::Renderer(RHIContext *rhi_context, Scene *scene, ResourceManager *resource_manager)
{
	m_impl                   = new Impl;
	m_impl->rhi_context      = rhi_context;
	m_impl->scene            = scene;
	m_impl->resource_manager = resource_manager;
	m_impl->shader_builder   = std::make_unique<ShaderBuilder>(rhi_context);

	// View
	{
		auto *view   = m_impl->black_board.Add<View>();
		view->buffer = m_impl->rhi_context->CreateBuffer<View::Info>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);
	}

	// GPU Scene
	{
		auto *gpu_scene = m_impl->black_board.Add<GPUScene>();
		gpu_scene->TLAS = m_impl->rhi_context->CreateAcccelerationStructure();
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
}

Renderer::~Renderer()
{
	delete m_impl;
}

void Renderer::Tick()
{
	m_impl->present_texture = nullptr;

	UpdateGPUScene();

	if (m_impl->render_graph)
	{
		m_impl->render_graph->Execute(m_impl->black_board);
	}
}

void Renderer::SetRenderGraph(std::unique_ptr<RenderGraph> &&render_graph)
{
	m_impl->rhi_context->WaitIdle();
	m_impl->render_graph    = std::move(render_graph);
	m_impl->present_texture = nullptr;
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

void Renderer::SetViewport(float width, float height)
{
	m_impl->viewport = glm::vec2{width, height};
}

glm::vec2 Renderer::GetViewport() const
{
	return m_impl->viewport;
}

void Renderer::SetPresentTexture(RHITexture *present_texture)
{
	m_impl->present_texture = present_texture;
}

RHITexture *Renderer::GetPresentTexture() const
{
	return m_impl->present_texture ? m_impl->present_texture : m_impl->black_board.Get<DummyTexture>()->black_opaque.get();
}

void Renderer::UpdateView(Cmpt::Camera *camera)
{
	if (camera)
	{
		auto *view = m_impl->black_board.Get<View>();

		View::Info view_info = {};

		view_info.position                   = camera->GetNode()->GetComponent<Cmpt::Transform>()->GetWorldTransform()[3];
		view_info.projection_matrix          = camera->GetProjectionMatrix();
		view_info.view_matrix                = camera->GetViewMatrix();
		view_info.inv_projection_matrix      = camera->GetInvProjectionMatrix();
		view_info.inv_view_matrix            = camera->GetInvViewMatrix();
		view_info.view_projection_matrix     = camera->GetViewProjectionMatrix();
		view_info.inv_view_projection_matrix = camera->GetInvViewProjectionMatrix();

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

// RHITexture *Renderer::GetDummyTexture(DummyTexture dummy) const
//{
//	return m_impl->dummy_textures.at(dummy).get();
// }
//
// RHIAccelerationStructure *Renderer::GetTLAS() const
//{
//	return m_impl->tlas.get();
// }
//
// void Renderer::DrawScene(RHICommand *cmd_buffer, RHIPipelineState *pipeline_state, RHIDescriptor *descriptor, bool mesh_shader)
//{
// }
//
// const SceneInfo &Renderer::GetSceneInfo() const
//{
//	return m_impl->scene_info;
// }

RHIShader *Renderer::RequireShader(const std::string &filename, const std::string &entry_point, RHIShaderStage stage, std::vector<std::string> &&macros, std::vector<std::string> &&includes, bool cuda, bool force_recompile)
{
	return m_impl->shader_builder->RequireShader(filename, entry_point, stage, std::move(macros), std::move(includes), cuda, force_recompile);
}

ShaderMeta Renderer::RequireShaderMeta(RHIShader *shader) const
{
	return m_impl->shader_builder->RequireShaderMeta(shader);
}

// RHIShader *Renderer::RequireMaterialShader(MaterialGraph *material_graph, const std::string &filename, const std::string &entry_point, RHIShaderStage stage, std::vector<std::string> &&macros, std::vector<std::string> &&includes)
//{
//	size_t material_hash = Hash(material_graph->GetDesc().name);
//	if (!Path::GetInstance().IsExist("bin/Materials/" + std::to_string(material_hash) + ".hlsli") || material_graph->Update())
//	{
//		MaterialGraphBuilder builder(m_impl->rhi_context);
//		builder.Compile(material_graph);
//	}
//
//	macros.push_back("MATERIAL_COMPILATION");
//	includes.push_back("bin/Materials/" + std::to_string(material_hash) + ".hlsli");
//
//	return RequireShader(filename, entry_point, stage, std::move(macros), std::move(includes), false, material_graph->Update());
// }

void Renderer::UpdateGPUScene()
{
	auto *gpu_scene = m_impl->black_board.Get<GPUScene>();

	// Update Mesh Buffer
	if (m_impl->resource_manager->Update<ResourceType::Mesh>())
	{
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

	// Update Skinned Mesh Buffer
	{
		if (m_impl->resource_manager->Update<ResourceType::SkinnedMesh>())
		{
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

	// Update 2D Textures
	if (m_impl->resource_manager->Update<ResourceType::Texture2D>())
	{
		gpu_scene->textures.texture_2d.clear();

		auto resources = m_impl->resource_manager->GetResources<ResourceType::Texture2D>();
		for (auto &resource : resources)
		{
			auto *texture2d = m_impl->resource_manager->Get<ResourceType::Texture2D>(resource);
			gpu_scene->textures.texture_2d.push_back(texture2d->GetTexture());
		}
	}

	// Update Mesh
	{
		std::vector<GPUScene::Instance> instances;

		auto meshes = m_impl->scene->GetComponents<Cmpt::MeshRenderer>();
		for (auto &mesh : meshes)
		{
			auto &submeshes = mesh->GetSubmeshes();
			for (auto &submesh : submeshes)
			{
				GPUScene::Instance instance = {};
				instance.transform          = mesh->GetNode()->GetComponent<Cmpt::Transform>()->GetWorldTransform();
				instance.mesh_id            = static_cast<uint32_t>(m_impl->resource_manager->Index<ResourceType::Mesh>(submesh));
				instances.push_back(instance);
			}
		}

		if (!instances.empty())
		{
			if (!gpu_scene->mesh_instances ||
			    gpu_scene->mesh_instances->GetDesc().size < instances.size() * sizeof(GPUScene::Instance))
			{
				gpu_scene->mesh_instances = m_impl->rhi_context->CreateBuffer<GPUScene::Instance>(instances.size(), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
			}

			gpu_scene->mesh_instances->CopyToDevice(instances.data(), instances.size() * sizeof(GPUScene::Instance));
		}
	}

	// Update Skinned Mesh
	{
		std::vector<GPUScene::Instance> instances;

		auto meshes = m_impl->scene->GetComponents<Cmpt::SkinnedMeshRenderer>();
		for (auto &mesh : meshes)
		{
			auto &submeshes = mesh->GetSubmeshes();
			for (auto &submesh : submeshes)
			{
				GPUScene::Instance instance = {};
				instance.transform          = mesh->GetNode()->GetComponent<Cmpt::Transform>()->GetWorldTransform();
				instance.mesh_id            = static_cast<uint32_t>(m_impl->resource_manager->Index<ResourceType::SkinnedMesh>(submesh));
				instances.push_back(instance);
			}
		}

		if (!instances.empty())
		{
			if (!gpu_scene->skinned_mesh_instances ||
			    gpu_scene->skinned_mesh_instances->GetDesc().size < instances.size() * sizeof(GPUScene::Instance))
			{
				gpu_scene->skinned_mesh_instances = m_impl->rhi_context->CreateBuffer<GPUScene::Instance>(instances.size(), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
			}

			gpu_scene->skinned_mesh_instances->CopyToDevice(instances.data(), instances.size() * sizeof(GPUScene::Instance));
		}
	}

	// m_scene_info.static_vertex_buffers.clear();
	// m_scene_info.static_index_buffers.clear();
	// m_scene_info.meshlet_vertex_buffers.clear();
	// m_scene_info.meshlet_primitive_buffers.clear();
	// m_scene_info.meshlet_buffers.clear();
	// m_scene_info.meshlet_count.clear();
	// m_scene_info.textures.clear();
	// m_scene_info.materials.clear();

	// std::vector<InstanceData> instances;
	// instances.reserve(p_scene->Size());

	//{
	//	size_t model_count = p_resource_manager->GetResourceValidUUID<ResourceType::Model>().size();
	//	if (m_scene_info.static_vertex_buffers.capacity() < model_count)
	//	{
	//		m_scene_info.static_vertex_buffers.reserve(model_count);
	//		m_scene_info.static_index_buffers.reserve(model_count);
	//		m_scene_info.meshlet_vertex_buffers.reserve(model_count);
	//		m_scene_info.meshlet_primitive_buffers.reserve(model_count);
	//		m_scene_info.meshlet_buffers.reserve(model_count);
	//	}
	//}

	//// Collect Model Info
	// for (auto &uuid : p_resource_manager->GetResourceValidUUID<ResourceType::Model>())
	//{
	//	auto *resource = p_resource_manager->GetResource<ResourceType::Model>(uuid);
	//	m_scene_info.static_vertex_buffers.push_back(resource->GetVertexBuffer());
	//	m_scene_info.static_index_buffers.push_back(resource->GetIndexBuffer());
	//	m_scene_info.meshlet_vertex_buffers.push_back(resource->GetMeshletVertexBuffer());
	//	m_scene_info.meshlet_primitive_buffers.push_back(resource->GetMeshletPrimitiveBuffer());
	//	m_scene_info.meshlet_buffers.push_back(resource->GetMeshletBuffer());
	// }

	//// Collect Texture Info
	//{
	//	size_t texture_count = p_resource_manager->GetResourceValidUUID<ResourceType::Texture>().size();
	//	m_scene_info.textures.reserve(texture_count);
	//}

	// for (auto &uuid : p_resource_manager->GetResourceValidUUID<ResourceType::Texture>())
	//{
	//	auto *resource = p_resource_manager->GetResource<ResourceType::Texture>(uuid);
	//	m_scene_info.textures.push_back(resource->GetTexture());
	// }

	//// Collect Material Info
	//{
	//	size_t material_count = p_resource_manager->GetResourceValidUUID<ResourceType::Material>().size();
	//	m_scene_info.materials.reserve(material_count);
	//}

	// for (auto &uuid : p_resource_manager->GetResourceValidUUID<ResourceType::Material>())
	//{
	//	auto *resource = p_resource_manager->GetResource<ResourceType::Material>(uuid);
	//	m_scene_info.materials.push_back(resource->Get());
	// }

	//// Collect Light Info
	//{
	//	struct LightData
	//	{
	//		glm::vec3 color;
	//		uint32_t  type;
	//		glm::vec3 position;
	//		float     intensity;
	//		glm::vec3 direction;
	//		float     range;
	//		alignas(16) float radius;
	//		float cut_off;
	//		float outer_cut_off;
	//	};
	//	std::vector<LightData> light_data;
	//	light_data.reserve(16);
	//	// Point Light
	//	p_scene->GroupExecute<PointLightComponent, TransformComponent>([&](uint32_t entity, PointLightComponent &light, TransformComponent &transform) {
	//		LightData data = {};
	//		data.type      = static_cast<uint32_t>(LightType::Point);
	//		data.color     = light.color;
	//		data.intensity = light.intensity;
	//		data.radius    = light.radius;
	//		data.position  = transform.world_transform[3];
	//		data.range     = light.range;
	//		light_data.push_back(data);
	//	});

	//	if (!light_data.empty())
	//	{
	//		if (!m_scene_info.light_buffer || light_data.size() * sizeof(LightData) != m_scene_info.light_buffer->GetDesc().size)
	//		{
	//			m_impl->rhi_context->WaitIdle();
	//			m_scene_info.light_buffer = m_impl->rhi_context->CreateBuffer<LightData>(light_data.size(), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
	//		}
	//		m_scene_info.light_buffer->CopyToDevice(light_data.data(), light_data.size() * sizeof(LightData), 0);
	//	}
	//	else if (!m_scene_info.light_buffer)
	//	{
	//		m_impl->rhi_context->WaitIdle();
	//		m_scene_info.light_buffer = m_impl->rhi_context->CreateBuffer<LightData>(1, RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
	//	}
	//}

	//// Update TLAS
	// TLASDesc desc = {};
	// desc.instances.reserve(p_scene->Size());
	// desc.name = p_scene->GetName();
	// p_scene->GroupExecute<StaticMeshComponent, TransformComponent>([&](uint32_t entity, StaticMeshComponent &static_mesh, TransformComponent &transform) {
	//	auto *resource = p_resource_manager->GetResource<ResourceType::Model>(static_mesh.uuid);
	//	if (resource)
	//	{
	//		for (uint32_t i = 0; i < resource->GetSubmeshes().size(); i++)
	//		{
	//			const Submesh &submesh = resource->GetSubmeshes()[i];

	//			TLASDesc::InstanceInfo instance_info = {};
	//			instance_info.transform              = transform.world_transform * submesh.pre_transform;
	//			instance_info.material_id            = 0;
	//			instance_info.blas                   = resource->GetBLAS(i);

	//			AABB aabb = resource->GetSubmeshes()[i].aabb.Transform(instance_info.transform);

	//			desc.instances.emplace_back(std::move(instance_info));

	//			InstanceData instance_data   = {};
	//			instance_data.aabb_max       = aabb.max;
	//			instance_data.aabb_min       = aabb.min;
	//			instance_data.model_id       = static_cast<uint32_t>(p_resource_manager->GetResourceIndex<ResourceType::Model>(static_mesh.uuid));
	//			instance_data.transform      = instance_info.transform;
	//			instance_data.material       = 0;
	//			instance_data.vertex_offset  = submesh.vertices_offset;
	//			instance_data.index_offset   = submesh.indices_offset;
	//			instance_data.meshlet_offset = submesh.meshlet_offset;
	//			instance_data.meshlet_count  = submesh.meshlet_count;
	//			instances.emplace_back(std::move(instance_data));
	//			m_scene_info.meshlet_count.push_back(resource->GetSubmeshes()[i].meshlet_count);
	//		}
	//	}
	//});

	// if (!desc.instances.empty())
	//{
	//	{
	//		if (!m_scene_info.instance_buffer || m_scene_info.instance_buffer->GetDesc().size != instances.size() * sizeof(InstanceData))
	//		{
	//			m_impl->rhi_context->WaitIdle();
	//			m_scene_info.instance_buffer = m_impl->rhi_context->CreateBuffer<InstanceData>(instances.size() * sizeof(InstanceData), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
	//		}
	//		m_scene_info.instance_buffer->CopyToDevice(instances.data(), instances.size() * sizeof(InstanceData), 0);
	//	}

	//	{
	//		auto *cmd_buffer = m_impl->rhi_context->CreateCommand(RHIQueueFamily::Compute);
	//		cmd_buffer->Begin();
	//		m_tlas->Update(cmd_buffer, desc);
	//		cmd_buffer->End();
	//		m_impl->rhi_context->Submit({cmd_buffer});
	//	}
	//}

	// m_scene_info.top_level_as = m_tlas.get();
}

}        // namespace Ilum