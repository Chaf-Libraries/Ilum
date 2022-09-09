#include "Renderer.hpp"

#include <CodeGeneration/Meta/RHIMeta.hpp>
#include <Core/Path.hpp>
#include <RenderCore/ShaderCompiler/ShaderCompiler.hpp>
#include <Scene/Scene.hpp>

namespace Ilum
{
Renderer::Renderer(RHIContext *rhi_context, Scene *scene, ResourceManager *resource_manager) :
    p_rhi_context(rhi_context), p_scene(scene), p_resource_manager(resource_manager)
{
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
		auto queue = p_rhi_context->CreateQueue(RHIQueueFamily::Graphics, 1);
		auto fence = p_rhi_context->CreateFence();
		queue->Submit({cmd_buffer});
		queue->Execute(fence.get());
		fence->Wait();
	}
}

Renderer::~Renderer()
{
}

void Renderer::Tick()
{
	m_present_texture = nullptr;

	if (m_render_graph)
	{
		m_render_graph->Execute();
	}
}

void Renderer::SetRenderGraph(std::unique_ptr<RenderGraph> &&render_graph)
{
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

RHIShader *Renderer::RequireShader(const std::string &filename, const std::string &entry_point, RHIShaderStage stage, const std::vector<std::string> &macros)
{
	size_t hash = Hash(filename, entry_point, stage, macros);

	if (m_shader_cache.find(hash) != m_shader_cache.end())
	{
		return m_shader_cache.at(hash).get();
	}

	std::string cache_path = "./bin/Shaders/" + std::to_string(hash) + ".shader";

	std::map<RHIBackend, std::vector<uint8_t>> shader_bin = {
	    {RHIBackend::Vulkan, {}},
	    {RHIBackend::DX12, {}},
	    {RHIBackend::CUDA, {}},
	};
	ShaderMeta meta;

	if (Path::GetInstance().IsExist(cache_path))
	{
		// Read from cache
		size_t last_write = 0;

		DESERIALIZE(
		    cache_path,
		    last_write,
		    shader_bin[RHIBackend::Vulkan],
		    shader_bin[RHIBackend::DX12],
		    shader_bin[RHIBackend::CUDA],
		    meta);

		if (last_write == std::filesystem::last_write_time(filename).time_since_epoch().count() && !shader_bin[p_rhi_context->GetBackend()].empty())
		{
			LOG_INFO("Load shader {} from cache", filename);
			std::unique_ptr<RHIShader> shader = p_rhi_context->CreateShader(entry_point, shader_bin[p_rhi_context->GetBackend()]);
			m_shader_meta_cache.emplace(shader.get(), std::move(meta));
			m_shader_cache.emplace(hash, std::move(shader));
			return m_shader_cache.at(hash).get();
		}
		LOG_INFO("Cache of shader {} is out of date, recompile it", filename);
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
		LOG_INFO("Compiling shader {}...", filename);
		shader_bin[p_rhi_context->GetBackend()] = ShaderCompiler::GetInstance().Compile(desc, meta);

		if (shader_bin[p_rhi_context->GetBackend()].empty())
		{
			LOG_ERROR("Shader {} compiled failed!", filename);
			return nullptr;
		}

		LOG_ERROR("Shader {} compiled successfully, caching it...", filename);

		SERIALIZE(
		    cache_path,
		    (size_t) std::filesystem::last_write_time(filename).time_since_epoch().count(),
		    shader_bin[RHIBackend::Vulkan],
		    shader_bin[RHIBackend::DX12],
		    shader_bin[RHIBackend::CUDA],
		    meta);

		std::unique_ptr<RHIShader> shader = p_rhi_context->CreateShader(entry_point, shader_bin[p_rhi_context->GetBackend()]);
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

}        // namespace Ilum