#include "Renderer.hpp"

#include <Core/Path.hpp>
#include <RHI/RHIContext.hpp>
#include <RenderCore/ShaderCompiler/ShaderCompiler.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Component/AllComponent.hpp>
#include <Scene/Entity.hpp>
#include <Scene/Scene.hpp>

#include <cereal/types/vector.hpp>

namespace Ilum
{
struct InstanceData
{
	glm::mat4 transform;

	glm::vec3 aabb_min;
	uint32_t  material;

	glm::vec3 aabb_max;
	uint32_t  model_id;

	uint32_t meshlet_count;
	uint32_t meshlet_offset;
	uint32_t vertex_offset;
	uint32_t index_offset;
};

Renderer::Renderer(RHIContext *rhi_context, Scene *scene, ResourceManager *resource_manager) :
    p_rhi_context(rhi_context), p_scene(scene), p_resource_manager(resource_manager)
{
	m_view_buffer = p_rhi_context->CreateBuffer<ViewInfo>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);
	m_tlas        = p_rhi_context->CreateAcccelerationStructure();

	// Create dummy texture
	{
		m_dummy_textures[DummyTexture::WhiteOpaque]      = p_rhi_context->CreateTexture2D(1, 1, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::Transfer | RHITextureUsage::ShaderResource, false);
		m_dummy_textures[DummyTexture::BlackOpaque]      = p_rhi_context->CreateTexture2D(1, 1, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::Transfer | RHITextureUsage::ShaderResource, false);
		m_dummy_textures[DummyTexture::WhiteTransparent] = p_rhi_context->CreateTexture2D(1, 1, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::Transfer | RHITextureUsage::ShaderResource, false);
		m_dummy_textures[DummyTexture::BlackTransparent] = p_rhi_context->CreateTexture2D(1, 1, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::Transfer | RHITextureUsage::ShaderResource, false);

		auto white_opaque_buffer      = p_rhi_context->CreateBuffer<glm::vec4>(1, RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU);
		auto black_opaque_buffer      = p_rhi_context->CreateBuffer<glm::vec4>(1, RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU);
		auto white_transparent_buffer = p_rhi_context->CreateBuffer<glm::vec4>(1, RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU);
		auto black_transparent_buffer = p_rhi_context->CreateBuffer<glm::vec4>(1, RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU);

		glm::vec4 white_opaque      = {1.f, 1.f, 1.f, 1.f};
		glm::vec4 black_opaque      = {0.f, 0.f, 0.f, 1.f};
		glm::vec4 white_transparent = {1.f, 1.f, 1.f, 0.f};
		glm::vec4 black_transparent = {0.f, 0.f, 0.f, 0.f};

		white_opaque_buffer->CopyToDevice(&white_opaque);
		black_opaque_buffer->CopyToDevice(&black_opaque);
		white_transparent_buffer->CopyToDevice(&white_transparent);
		black_transparent_buffer->CopyToDevice(&black_transparent);

		auto *cmd_buffer = p_rhi_context->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->ResourceStateTransition({TextureStateTransition{m_dummy_textures[DummyTexture::WhiteOpaque].get(), RHIResourceState::Undefined, RHIResourceState::TransferDest, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}},
		                                     TextureStateTransition{m_dummy_textures[DummyTexture::BlackOpaque].get(), RHIResourceState::Undefined, RHIResourceState::TransferDest, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}},
		                                     TextureStateTransition{m_dummy_textures[DummyTexture::WhiteTransparent].get(), RHIResourceState::Undefined, RHIResourceState::TransferDest, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}},
		                                     TextureStateTransition{m_dummy_textures[DummyTexture::BlackTransparent].get(), RHIResourceState::Undefined, RHIResourceState::TransferDest, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
		                                    {});
		cmd_buffer->CopyBufferToTexture(white_opaque_buffer.get(), m_dummy_textures[DummyTexture::WhiteOpaque].get(), 0, 0, 1);
		cmd_buffer->CopyBufferToTexture(black_opaque_buffer.get(), m_dummy_textures[DummyTexture::BlackOpaque].get(), 0, 0, 1);
		cmd_buffer->CopyBufferToTexture(white_transparent_buffer.get(), m_dummy_textures[DummyTexture::WhiteTransparent].get(), 0, 0, 1);
		cmd_buffer->CopyBufferToTexture(black_transparent_buffer.get(), m_dummy_textures[DummyTexture::BlackTransparent].get(), 0, 0, 1);
		cmd_buffer->ResourceStateTransition({TextureStateTransition{m_dummy_textures[DummyTexture::WhiteOpaque].get(), RHIResourceState::TransferDest, RHIResourceState::ShaderResource, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}},
		                                     TextureStateTransition{m_dummy_textures[DummyTexture::BlackOpaque].get(), RHIResourceState::TransferDest, RHIResourceState::ShaderResource, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}},
		                                     TextureStateTransition{m_dummy_textures[DummyTexture::WhiteTransparent].get(), RHIResourceState::TransferDest, RHIResourceState::ShaderResource, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}},
		                                     TextureStateTransition{m_dummy_textures[DummyTexture::BlackTransparent].get(), RHIResourceState::TransferDest, RHIResourceState::ShaderResource, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
		                                    {});
		cmd_buffer->End();
		p_rhi_context->Execute(cmd_buffer);
	}
}

Renderer::~Renderer()
{
}

void Renderer::Tick()
{
	m_present_texture = nullptr;

	p_resource_manager->Tick();

	UpdateScene();

	if (m_render_graph)
	{
		m_render_graph->Execute();
	}
}

void Renderer::SetRenderGraph(std::unique_ptr<RenderGraph> &&render_graph)
{
	p_rhi_context->WaitIdle();
	m_render_graph    = std::move(render_graph);
	m_present_texture = nullptr;
}

RenderGraph *Renderer::GetRenderGraph() const
{
	return m_render_graph.get();
}

RHIContext *Renderer::GetRHIContext() const
{
	return p_rhi_context;
}

ResourceManager *Renderer::GetResourceManager() const
{
	return p_resource_manager;
}

void Renderer::SetViewport(float width, float height)
{
	m_viewport = glm::vec2{width, height};
}

glm::vec2 Renderer::GetViewport() const
{
	return m_viewport;
}

void Renderer::SetPresentTexture(RHITexture *present_texture)
{
	m_present_texture = present_texture;
}

RHITexture *Renderer::GetPresentTexture() const
{
	return m_present_texture ? m_present_texture : m_dummy_textures.at(DummyTexture::BlackOpaque).get();
}

void Renderer::SetViewInfo(const ViewInfo &view_info)
{
	m_view_buffer->CopyToDevice(&view_info);
}

RHIBuffer *Renderer::GetViewBuffer() const
{
	return m_view_buffer.get();
}

Scene *Renderer::GetScene() const
{
	return p_scene;
}

void Renderer::Reset()
{
	m_shader_cache.clear();
	m_shader_meta_cache.clear();
}

RHITexture *Renderer::GetDummyTexture(DummyTexture dummy) const
{
	return m_dummy_textures.at(dummy).get();
}

RHIAccelerationStructure *Renderer::GetTLAS() const
{
	return m_tlas.get();
}

void Renderer::DrawScene(RHICommand *cmd_buffer, RHIPipelineState *pipeline_state, RHIDescriptor *descriptor, bool mesh_shader)
{
}

const SceneInfo &Renderer::GetSceneInfo() const
{
	return m_scene_info;
}

RHIShader *Renderer::RequireShader(const std::string &filename, const std::string &entry_point, RHIShaderStage stage, const std::vector<std::string> &macros, bool cuda)
{
	size_t hash = Hash(filename, entry_point, stage, macros, cuda);

	if (!Path::GetInstance().IsExist("./bin/Shaders"))
	{
		Path::GetInstance().CreatePath("./bin/Shaders");
	}

	if (m_shader_cache.find(hash) != m_shader_cache.end())
	{
		return m_shader_cache.at(hash).get();
	}

	std::string cache_path = "./bin/Shaders/" + std::to_string(hash) + ".shader";

	std::vector<uint8_t> shader_bin;
	ShaderMeta           meta;

	if (Path::GetInstance().IsExist(cache_path))
	{
		// Read from cache
		size_t last_write = 0;

		DESERIALIZE(
		    cache_path,
		    last_write,
		    shader_bin,
		    meta);

		if (last_write == std::filesystem::last_write_time(filename).time_since_epoch().count() && !shader_bin.empty())
		{
			LOG_INFO("Load shader {} with entry point \"{}\" from cache", filename, entry_point);
			std::unique_ptr<RHIShader> shader = p_rhi_context->CreateShader(entry_point, shader_bin, cuda);
			m_shader_meta_cache.emplace(shader.get(), std::move(meta));
			m_shader_cache.emplace(hash, std::move(shader));
			return m_shader_cache.at(hash).get();
		}
		LOG_INFO("Cache of shader {} with entry point \"{}\" is out of date, recompile it", filename, entry_point);
	}

	{
		std::vector<uint8_t> shader_code;
		Path::GetInstance().Read(filename, shader_code);

		ShaderDesc desc = {};
		desc.code.resize(shader_code.size());
		std::memcpy(desc.code.data(), shader_code.data(), shader_code.size());
		desc.source      = Path::GetInstance().GetFileExtension(filename) == ".hlsl" ? ShaderSource::HLSL : ShaderSource::GLSL;
		desc.stage       = stage;
		desc.entry_point = entry_point;
		desc.macros      = macros;
		if (cuda)
		{
			desc.target = ShaderTarget::PTX;
		}
		else
		{
			switch (p_rhi_context->GetBackend())
			{
				case RHIBackend::Vulkan:
					desc.target = ShaderTarget::SPIRV;
					break;
				case RHIBackend::DX12:
					desc.target = ShaderTarget::DXIL;
					break;
				case RHIBackend::CUDA:
					desc.target = ShaderTarget::PTX;
					break;
				default:
					break;
			}
		}

		LOG_INFO("Compiling shader {} with entry point \"{}\"...", filename, entry_point);
		shader_bin = ShaderCompiler::GetInstance().Compile(desc, meta);

		if (shader_bin.empty())
		{
			LOG_ERROR("Shader {} with entry point \"{}\" compiled failed!", filename, entry_point);
			return nullptr;
		}

		LOG_ERROR("Shader {} with entry point \"{}\" compiled successfully, caching it...", filename, entry_point);

		SERIALIZE(
		    cache_path,
		    (size_t) std::filesystem::last_write_time(filename).time_since_epoch().count(),
		    shader_bin,
		    meta);

		std::unique_ptr<RHIShader> shader = p_rhi_context->CreateShader(entry_point, shader_bin, cuda);
		m_shader_meta_cache.emplace(shader.get(), std::move(meta));
		m_shader_cache.emplace(hash, std::move(shader));
		return m_shader_cache.at(hash).get();
	}

	return nullptr;
}

ShaderMeta Renderer::RequireShaderMeta(RHIShader *shader) const
{
	return m_shader_meta_cache.at(shader);
}

void Renderer::UpdateScene()
{
	m_scene_info.static_vertex_buffers.clear();
	m_scene_info.static_index_buffers.clear();
	m_scene_info.meshlet_vertex_buffers.clear();
	m_scene_info.meshlet_primitive_buffers.clear();
	m_scene_info.meshlet_buffers.clear();
	m_scene_info.meshlet_count.clear();

	std::vector<InstanceData> instances;
	instances.reserve(p_scene->Size());

	{
		size_t model_count = p_resource_manager->GetResourceValidUUID<ResourceType::Model>().size();
		if (m_scene_info.static_vertex_buffers.capacity() < model_count)
		{
			m_scene_info.static_vertex_buffers.reserve(model_count);
			m_scene_info.static_index_buffers.reserve(model_count);
			m_scene_info.meshlet_vertex_buffers.reserve(model_count);
			m_scene_info.meshlet_primitive_buffers.reserve(model_count);
			m_scene_info.meshlet_buffers.reserve(model_count);
		}
	}

	// Collect Model Info
	for (auto &uuid : p_resource_manager->GetResourceValidUUID<ResourceType::Model>())
	{
		auto *resource = p_resource_manager->GetResource<ResourceType::Model>(uuid);
		m_scene_info.static_vertex_buffers.push_back(resource->GetVertexBuffer());
		m_scene_info.static_index_buffers.push_back(resource->GetIndexBuffer());
		m_scene_info.meshlet_vertex_buffers.push_back(resource->GetMeshletVertexBuffer());
		m_scene_info.meshlet_primitive_buffers.push_back(resource->GetMeshletPrimitiveBuffer());
		m_scene_info.meshlet_buffers.push_back(resource->GetMeshletBuffer());
	}

	// Collect Texture Info
	{
		size_t texture_count = p_resource_manager->GetResourceValidUUID<ResourceType::Texture>().size();
		m_scene_info.textures.reserve(texture_count);
	}

	for (auto &uuid : p_resource_manager->GetResourceValidUUID<ResourceType::Texture>())
	{
		auto *resource = p_resource_manager->GetResource<ResourceType::Texture>(uuid);
		m_scene_info.textures.push_back(resource->GetTexture());
	}

	// Update TLAS
	TLASDesc desc = {};
	desc.instances.reserve(p_scene->Size());
	desc.name = p_scene->GetName();
	p_scene->GroupExecute<StaticMeshComponent, TransformComponent>([&](uint32_t entity, StaticMeshComponent &static_mesh, TransformComponent &transform) {
		auto *resource = p_resource_manager->GetResource<ResourceType::Model>(static_mesh.uuid);
		if (resource)
		{
			for (uint32_t i = 0; i < resource->GetSubmeshes().size(); i++)
			{
				const Submesh &submesh = resource->GetSubmeshes()[i];

				TLASDesc::InstanceInfo instance_info = {};
				instance_info.transform              = transform.world_transform * submesh.pre_transform;
				instance_info.material_id            = 0;
				instance_info.blas                   = resource->GetBLAS(i);

				AABB aabb = resource->GetSubmeshes()[i].aabb.Transform(instance_info.transform);

				desc.instances.emplace_back(std::move(instance_info));

				InstanceData instance_data   = {};
				instance_data.aabb_max       = aabb.max;
				instance_data.aabb_min       = aabb.min;
				instance_data.model_id       = static_cast<uint32_t>(p_resource_manager->GetResourceIndex<ResourceType::Model>(static_mesh.uuid));
				instance_data.transform      = instance_info.transform;
				instance_data.material       = 0;
				instance_data.vertex_offset  = submesh.vertices_offset;
				instance_data.index_offset  = submesh.indices_offset;
				instance_data.meshlet_offset = submesh.meshlet_offset;
				instance_data.meshlet_count  = submesh.meshlet_count;
				instances.emplace_back(std::move(instance_data));
				m_scene_info.meshlet_count.push_back(resource->GetSubmeshes()[i].meshlet_count);
			}
		}
	});

	if (!desc.instances.empty())
	{
		{
			if (!m_scene_info.instance_buffer || m_scene_info.instance_buffer->GetDesc().size != instances.size() * sizeof(InstanceData))
			{
				m_scene_info.instance_buffer = p_rhi_context->CreateBuffer<InstanceData>(instances.size() * sizeof(InstanceData), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
			}
			m_scene_info.instance_buffer->CopyToDevice(instances.data(), instances.size() * sizeof(InstanceData), 0);
		}

		{
			auto *cmd_buffer = p_rhi_context->CreateCommand(RHIQueueFamily::Compute);
			cmd_buffer->Begin();
			m_tlas->Update(cmd_buffer, desc);
			cmd_buffer->End();
			p_rhi_context->Submit({cmd_buffer});
		}
	}

	m_scene_info.top_level_as = m_tlas.get();
}

}        // namespace Ilum