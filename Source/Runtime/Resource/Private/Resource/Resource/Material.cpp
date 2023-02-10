#include "Resource/Material.hpp"
#include "Resource/Mesh.hpp"
#include "Resource/Texture2D.hpp"
#include "ResourceManager.hpp"

#include <Material/MaterialCompiler.hpp>
#include <Material/MaterialData.hpp>
#include <ShaderCompiler/ShaderCompiler.hpp>

#include <mustache.hpp>

namespace Ilum
{
struct Resource<ResourceType::Material>::Impl
{
	MaterialGraphDesc desc;

	MaterialCompilationContext context;

	std::string layout;

	MaterialData data;

	bool valid = false;

	bool dirty = false;
};

Resource<ResourceType::Material>::Resource(RHIContext *rhi_context, const std::string &name) :
    IResource(rhi_context, name, ResourceType::Material)
{
}

Resource<ResourceType::Material>::Resource(RHIContext *rhi_context, const std::string &name, MaterialGraphDesc &&desc) :
    IResource(name)
{
	m_impl = new Impl;

	m_impl->desc = std::move(desc);

	std::vector<uint8_t> thumbnail_data;

	SERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Material), thumbnail_data, m_impl->desc, m_impl->layout, m_impl->context, m_impl->data);
}

Resource<ResourceType::Material>::~Resource()
{
	delete m_impl;
}

bool Resource<ResourceType::Material>::Validate() const
{
	return m_impl != nullptr;
}

void Resource<ResourceType::Material>::Load(RHIContext *rhi_context)
{
	m_impl = new Impl;

	std::vector<uint8_t> thumbnail_data;
	DESERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Material), thumbnail_data, m_impl->desc, m_impl->layout, m_impl->context, m_impl->data);

	m_impl->valid = Path::GetInstance().IsExist(fmt::format("Asset/Material/{}", m_impl->data.shader));
}

void Resource<ResourceType::Material>::Compile(RHIContext *rhi_context, ResourceManager *manager, RHITexture *dummy_texture, const std::string &layout)
{
	if (!layout.empty())
	{
		m_impl->layout = layout;
	}

	m_impl->valid = false;

	m_impl->context.Reset();
	m_impl->data.Reset();

	for (auto &[node_handle, node] : m_impl->desc.GetNodes())
	{
		node.EmitHLSL(m_impl->desc, manager, &m_impl->context);
	}

	if (!m_impl->context.output.bsdf.empty())
	{
		std::vector<uint8_t> shader_data;
		Path::GetInstance().Read("Source/Shaders/Material.hlsli", shader_data);
		std::string shader(shader_data.begin(), shader_data.end());

		kainjow::mustache::mustache mustache = {shader};
		kainjow::mustache::data     mustache_data{kainjow::mustache::data::type::object};

		{
			kainjow::mustache::data initializations{kainjow::mustache::data::type::list};
			kainjow::mustache::data textures{kainjow::mustache::data::type::list};
			kainjow::mustache::data samplers{kainjow::mustache::data::type::list};
			for (auto &variable : m_impl->context.variables)
			{
				initializations << kainjow::mustache::data{"Initialization", variable};
			}
			for (auto &[texture, texture_name] : m_impl->context.textures)
			{
				textures << kainjow::mustache::data{"Texture", texture};
			}
			for (auto &[sampler, desc] : m_impl->context.samplers)
			{
				samplers << kainjow::mustache::data{"Sampler", sampler};
			}
			for (auto &bsdf : m_impl->context.bsdfs)
			{
				if (bsdf.name != m_impl->context.output.bsdf)
				{
					initializations << kainjow::mustache::data{"Initialization", fmt::format("{} {};", bsdf.type, bsdf.name)};
					initializations << kainjow::mustache::data{"Initialization", fmt::format("{}", bsdf.initialization)};
				}
				else
				{
					mustache_data.set("BxDFType", bsdf.type);
					mustache_data.set("BxDFName", bsdf.name);
				}
			}
			initializations << kainjow::mustache::data{"Initialization", m_impl->context.bsdfs.back().initialization};

			mustache_data.set("Initializations", initializations);
			mustache_data.set("Textures", textures);
			mustache_data.set("Samplers", samplers);
		}

		shader = mustache.render(mustache_data);
		shader = std::string(shader.c_str());

		m_impl->data.signature = fmt::format("Signature_{}", Hash(shader));

		shader_data.resize(shader.length());
		std::memcpy(shader_data.data(), shader.data(), shader_data.size());

		m_impl->data.shader = fmt::format("{}.material.hlsli", m_impl->desc.GetName());

		Path::GetInstance().Save(fmt::format("Asset/Material/{}", m_impl->data.shader), shader_data);
	}

	m_impl->valid = true;
	m_impl->dirty = true;

	Update(rhi_context, manager, dummy_texture);
}

void Resource<ResourceType::Material>::Update(RHIContext *rhi_context, ResourceManager *manager, RHITexture *dummy_texture)
{
	if (!m_impl->valid)
	{
		Compile(rhi_context, manager, dummy_texture);
	}
	else
	{
		manager->SetDirty<ResourceType::Material>();

		m_impl->data.textures.clear();
		for (auto &[texture, texture_name] : m_impl->context.textures)
		{
			m_impl->data.textures.push_back(static_cast<uint32_t>(manager->Index<ResourceType::Texture2D>(texture_name)));
		}

		m_impl->data.samplers.clear();
		for (auto &[sampler, desc] : m_impl->context.samplers)
		{
			m_impl->data.samplers.push_back(rhi_context->GetSamplerIndex(desc));
		}
	}
}

void Resource<ResourceType::Material>::PostUpdate(RHIContext *rhi_context, const std::vector<RHITexture *> &scene_texture_2d, const std::vector<RHISampler *> &samplers, RHIBuffer *material_buffers, RHIBuffer *material_offsets)
{
	if (m_impl->dirty)
	{
		std::vector<uint8_t> thumbnail_data = RenderPreview(rhi_context, scene_texture_2d, samplers, material_buffers, material_offsets);
		SERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Material), thumbnail_data, m_impl->desc, m_impl->layout, m_impl->context, m_impl->data);
		m_impl->dirty = false;
	}
}

const MaterialData &Resource<ResourceType::Material>::GetMaterialData() const
{
	return m_impl->data;
}

const MaterialCompilationContext &Resource<ResourceType::Material>::GetCompilationContext() const
{
	return m_impl->context;
}

const std::string &Resource<ResourceType::Material>::GetLayout() const
{
	return m_impl->layout;
}

MaterialGraphDesc &Resource<ResourceType::Material>::GetDesc()
{
	return m_impl->desc;
}

bool Resource<ResourceType::Material>::IsValid() const
{
	return m_impl->valid;
}

std::vector<uint8_t> Resource<ResourceType::Material>::RenderPreview(RHIContext *rhi_context, const std::vector<RHITexture *> &scene_texture_2d, const std::vector<RHISampler *> &samplers, RHIBuffer *material_buffers, RHIBuffer *material_offsets)
{
	std::vector<Resource<ResourceType::Mesh>::Vertex> vertices;

	std::vector<uint32_t> indices;

	DESERIALIZE("Asset/BuildIn/MaterialBall.asset", vertices, indices);

	auto vertex_buffer = rhi_context->CreateBuffer<Resource<ResourceType::Mesh>::Vertex>(vertices.size(), RHIBufferUsage::Vertex, RHIMemoryUsage::CPU_TO_GPU);
	auto index_buffer  = rhi_context->CreateBuffer<uint32_t>(indices.size(), RHIBufferUsage::Index, RHIMemoryUsage::CPU_TO_GPU);

	vertex_buffer->CopyToDevice(vertices.data(), vertices.size() * sizeof(Resource<ResourceType::Mesh>::Vertex));
	index_buffer->CopyToDevice(indices.data(), indices.size() * sizeof(uint32_t));

	std::vector<uint8_t> raw_shader;
	Path::GetInstance().Read("./Source/Shaders/Preview/Material.hlsl", raw_shader);

	std::string shader_source;
	shader_source.resize(raw_shader.size());
	std::memcpy(shader_source.data(), raw_shader.data(), raw_shader.size());
	shader_source += "\n";

	ShaderDesc vertex_shader_desc  = {};
	vertex_shader_desc.path        = "./Source/Shaders/Preview/Material.hlsl";
	vertex_shader_desc.entry_point = "VSmain";
	vertex_shader_desc.stage       = RHIShaderStage::Vertex;
	vertex_shader_desc.source      = ShaderSource::HLSL;
	vertex_shader_desc.target      = ShaderTarget::SPIRV;
	vertex_shader_desc.code        = fmt::format("#include \"{}\"\n", m_impl->data.shader) + shader_source;
	vertex_shader_desc.macros      = {"USE_MATERIAL", m_impl->data.signature};

	ShaderDesc fragment_shader_desc  = {};
	fragment_shader_desc.path        = "./Source/Shaders/Preview/Material.hlsl";
	fragment_shader_desc.entry_point = "PSmain";
	fragment_shader_desc.stage       = RHIShaderStage::Fragment;
	fragment_shader_desc.source      = ShaderSource::HLSL;
	fragment_shader_desc.target      = ShaderTarget::SPIRV;
	fragment_shader_desc.code        = fmt::format("#include \"{}\"\n", m_impl->data.shader) + shader_source;
	fragment_shader_desc.macros      = {"USE_MATERIAL", m_impl->data.signature};

	ShaderMeta vertex_meta   = {};
	ShaderMeta fragment_meta = {};

	auto vertex_shader_spirv   = ShaderCompiler::GetInstance().Compile(vertex_shader_desc, vertex_meta);
	auto fragment_shader_spirv = ShaderCompiler::GetInstance().Compile(fragment_shader_desc, fragment_meta);

	auto vertex_shader   = rhi_context->CreateShader("VSmain", vertex_shader_spirv);
	auto fragment_shader = rhi_context->CreateShader("PSmain", fragment_shader_spirv);

	ShaderMeta shader_meta = vertex_meta;
	shader_meta += fragment_meta;

	BlendState blend_state = {};
	blend_state.attachment_states.resize(1);

	DepthStencilState depth_stencil_state  = {};
	depth_stencil_state.depth_test_enable  = true;
	depth_stencil_state.depth_write_enable = true;

	auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
	cmd_buffer->Begin();

	if (!m_thumbnail)
	{
		m_thumbnail = rhi_context->CreateTexture2D(128, 128, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::ShaderResource | RHITextureUsage::Transfer | RHITextureUsage::RenderTarget, false);
		cmd_buffer->ResourceStateTransition({
		                                        TextureStateTransition{m_thumbnail.get(), RHIResourceState::Undefined, RHIResourceState::RenderTarget},
		                                    },
		                                    {});
	}
	else
	{
		cmd_buffer->ResourceStateTransition({
		                                        TextureStateTransition{m_thumbnail.get(), RHIResourceState::ShaderResource, RHIResourceState::RenderTarget},
		                                    },
		                                    {});
	}

	auto depth_buffer   = rhi_context->CreateTexture2D(128, 128, RHIFormat::D32_FLOAT, RHITextureUsage::RenderTarget, false);
	auto uniform_buffer = rhi_context->CreateBuffer<glm::mat4>(2, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);
	auto staging_buffer = rhi_context->CreateBuffer(128ull * 128ull * 4ull * sizeof(uint8_t), RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_TO_CPU);
	auto render_target  = rhi_context->CreateRenderTarget();
	auto pipeline_state = rhi_context->CreatePipelineState();

	pipeline_state->SetBlendState(blend_state);
	pipeline_state->SetDepthStencilState(depth_stencil_state);
	pipeline_state->SetShader(RHIShaderStage::Vertex, vertex_shader.get());
	pipeline_state->SetShader(RHIShaderStage::Fragment, fragment_shader.get());
	pipeline_state->SetVertexInputState(VertexInputState{
	    {
	        VertexInputState::InputAttribute{RHIVertexSemantics::Position, 0, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, position)},
	        VertexInputState::InputAttribute{RHIVertexSemantics::Normal, 1, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, normal)},
	        VertexInputState::InputAttribute{RHIVertexSemantics::Texcoord, 3, 0, RHIFormat::R32G32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, texcoord0)},
	    },
	    {
	        VertexInputState::InputBinding{0, sizeof(Resource<ResourceType::Mesh>::Vertex), RHIVertexInputRate::Vertex},
	    }});

	{
		glm::vec3 center = glm::vec3(0.f);

		float radius = 4.f;
		float theta  = 0.f;
		float phi    = 60.f;

		glm::vec3 position  = center + radius * glm::vec3(glm::sin(glm::radians(phi)) * glm::sin(glm::radians(theta)), glm::cos(glm::radians(phi)), glm::sin(glm::radians(phi)) * glm::cos(glm::radians(theta)));
		glm::vec3 direction = glm::normalize(center - position);
		glm::vec3 right     = glm::normalize(glm::cross(direction, glm::vec3{0.f, 1.f, 0.f}));
		glm::vec3 up        = glm::normalize(glm::cross(right, direction));
		glm::mat4 transform = glm::perspective(glm::radians(45.f), 1.f, 0.01f, 1000.f) * glm::lookAt(position, center, up);
		glm::mat4 model     = glm::mat4_cast(glm::qua<float>(glm::radians(glm::vec3(135.f, 0.f, 180.f))));

		uniform_buffer->CopyToDevice(glm::value_ptr(transform), sizeof(transform));
		uniform_buffer->CopyToDevice(glm::value_ptr(model), sizeof(model), sizeof(glm::mat4));
	}

	auto descriptor = rhi_context->CreateDescriptor(shader_meta);
	descriptor->BindBuffer("UniformBuffer", uniform_buffer.get())
	    .BindTexture("Textures", scene_texture_2d, RHITextureDimension::Texture2D)
	    .BindSampler("Samplers", samplers)
	    .BindBuffer("MaterialOffsets", material_offsets)
	    .BindBuffer("MaterialBuffer", material_buffers);

	render_target->Set(0, m_thumbnail.get(), TextureRange{}, ColorAttachment{RHILoadAction::Clear, RHIStoreAction::Store, {0.1f, 0.1f, 0.1f, 1.f}});
	render_target->Set(depth_buffer.get(), TextureRange{}, DepthStencilAttachment{});

	cmd_buffer->ResourceStateTransition({
	                                        TextureStateTransition{depth_buffer.get(), RHIResourceState::Undefined, RHIResourceState::DepthWrite},
	                                    },
	                                    {});
	cmd_buffer->BeginRenderPass(render_target.get());
	cmd_buffer->SetViewport(128, 128);
	cmd_buffer->SetScissor(128, 128);
	cmd_buffer->BindDescriptor(descriptor);
	cmd_buffer->BindPipelineState(pipeline_state.get());
	cmd_buffer->BindVertexBuffer(0, vertex_buffer.get());
	cmd_buffer->BindIndexBuffer(index_buffer.get());
	cmd_buffer->DrawIndexed(static_cast<uint32_t>(indices.size()));
	cmd_buffer->EndRenderPass();
	cmd_buffer->ResourceStateTransition({
	                                        TextureStateTransition{m_thumbnail.get(), RHIResourceState::RenderTarget, RHIResourceState::TransferSource},
	                                    },
	                                    {});
	cmd_buffer->CopyTextureToBuffer(m_thumbnail.get(), staging_buffer.get(), 0, 0, 1);
	cmd_buffer->ResourceStateTransition({
	                                        TextureStateTransition{m_thumbnail.get(), RHIResourceState::TransferSource, RHIResourceState::ShaderResource},
	                                    },
	                                    {});
	cmd_buffer->End();
	rhi_context->Execute(cmd_buffer);

	std::vector<uint8_t> staging_data(staging_buffer->GetDesc().size);
	staging_buffer->CopyToHost(staging_data.data(), staging_buffer->GetDesc().size);

	return staging_data;
}
}        // namespace Ilum