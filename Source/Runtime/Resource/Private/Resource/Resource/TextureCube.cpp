#include "Resource/TextureCube.hpp"

#include <RHI/RHIContext.hpp>
#include <ShaderCompiler/ShaderCompiler.hpp>

#include <fstream>

namespace Ilum
{
struct Resource<ResourceType::TextureCube>::Impl
{
	std::unique_ptr<RHITexture> texture = nullptr;
};

Resource<ResourceType::TextureCube>::Resource(RHIContext *rhi_context, const std::string &name) :
    IResource(rhi_context, name, ResourceType::TextureCube)
{
}

Resource<ResourceType::TextureCube>::Resource(RHIContext *rhi_context, std::vector<uint8_t> &&data, const TextureDesc &desc) :
    IResource(desc.name)
{
	m_impl = std::make_unique<Impl>();

	auto cubemap2d = rhi_context->CreateTexture(desc);

	TextureDesc cubemap_desc = desc;
	cubemap_desc.width       = 512;
	cubemap_desc.height      = 512;
	cubemap_desc.depth       = 1;
	cubemap_desc.layers      = 6;
	cubemap_desc.format      = RHIFormat::R32G32B32A32_FLOAT;
	cubemap_desc.usage |= RHITextureUsage::RenderTarget;

	m_impl->texture = rhi_context->CreateTexture(cubemap_desc);
	m_thumbnail     = rhi_context->CreateTextureCube(128, 128, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::ShaderResource | RHITextureUsage::Transfer, false);

	auto cubemap_buffer = rhi_context->CreateBuffer(4ull * 1024ull * 1024ull * 4ull * 6ull, RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_TO_CPU);
	auto staging_buffer = rhi_context->CreateBuffer(glm::max(data.size(), 4ull * 128ull * 128ull), RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU);
	staging_buffer->CopyToDevice(data.data(), staging_buffer->GetDesc().size);

	// Convert equirectangular to cubemap
	//std::vector<uint8_t> raw_shader;
	//Path::GetInstance().Read("./Source/Shaders/PreProcess/EquirectangularToCubemap.hlsl", raw_shader);

	//std::string shader_source;
	//shader_source.resize(raw_shader.size());
	//std::memcpy(shader_source.data(), raw_shader.data(), raw_shader.size());
	//shader_source += "\n";

	//ShaderDesc vertex_shader_desc  = {};
	//vertex_shader_desc.path = "./Source/Shaders/PreProcess/EquirectangularToCubemap.hlsl";
	//vertex_shader_desc.entry_point = "VSmain";
	//vertex_shader_desc.stage       = RHIShaderStage::Vertex;
	//vertex_shader_desc.source      = ShaderSource::HLSL;
	//vertex_shader_desc.target      = ShaderTarget::SPIRV;
	//vertex_shader_desc.code        = shader_source;

	//ShaderDesc fragment_shader_desc  = {};
	//fragment_shader_desc.path = "./Source/Shaders/PreProcess/EquirectangularToCubemap.hlsl";
	//fragment_shader_desc.entry_point = "PSmain";
	//fragment_shader_desc.stage       = RHIShaderStage::Fragment;
	//fragment_shader_desc.source      = ShaderSource::HLSL;
	//fragment_shader_desc.target      = ShaderTarget::SPIRV;
	//fragment_shader_desc.code        = shader_source;

	//ShaderMeta vertex_meta   = {};
	//ShaderMeta fragment_meta = {};

	//auto vertex_shader_spirv   = ShaderCompiler::GetInstance().Compile(vertex_shader_desc, vertex_meta);
	//auto fragment_shader_spirv = ShaderCompiler::GetInstance().Compile(fragment_shader_desc, fragment_meta);

	//auto vertex_shader   = rhi_context->CreateShader("VSmain", vertex_shader_spirv);
	//auto fragment_shader = rhi_context->CreateShader("PSmain", fragment_shader_spirv);

	//ShaderMeta shader_meta = vertex_meta;
	//shader_meta += fragment_meta;

	//SERIALIZE("Asset/BuildIn/equirectangular_to_cubemap.shader.asset", vertex_shader_spirv, fragment_shader_spirv, shader_meta);

	std::vector<uint8_t> vertex_shader_spirv, fragment_shader_spirv;
	ShaderMeta           shader_meta;

	DESERIALIZE("Asset/BuildIn/equirectangular_to_cubemap.shader.asset", vertex_shader_spirv, fragment_shader_spirv, shader_meta);

	auto vertex_shader   = rhi_context->CreateShader("VSmain", vertex_shader_spirv);
	auto fragment_shader = rhi_context->CreateShader("PSmain", fragment_shader_spirv);

	BlendState blend_state = {};
	blend_state.attachment_states.resize(1);

	DepthStencilState depth_stencil_state  = {};
	depth_stencil_state.depth_test_enable  = false;
	depth_stencil_state.depth_write_enable = false;

	auto pipeline_state = rhi_context->CreatePipelineState();
	auto descriptor     = rhi_context->CreateDescriptor(shader_meta);
	auto render_target  = rhi_context->CreateRenderTarget();

	render_target->Set(0, m_impl->texture.get(), TextureRange{RHITextureDimension::Texture2DArray, 0, 1, 0, 6}, ColorAttachment{});

	pipeline_state->SetShader(RHIShaderStage::Vertex, vertex_shader.get());
	pipeline_state->SetShader(RHIShaderStage::Fragment, fragment_shader.get());
	pipeline_state->SetBlendState(blend_state);
	pipeline_state->SetDepthStencilState(depth_stencil_state);

	descriptor->BindTexture("InputTexture", cubemap2d.get(), RHITextureDimension::Texture2D)
	    .BindSampler("TexSampler", rhi_context->CreateSampler(SamplerDesc::LinearClamp()));

	auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
	cmd_buffer->Begin();
	cmd_buffer->ResourceStateTransition(
	    {TextureStateTransition{
	         cubemap2d.get(),
	         RHIResourceState::Undefined,
	         RHIResourceState::TransferDest,
	         TextureRange{RHITextureDimension::Texture2D, 0, cubemap2d.get()->GetDesc().mips, 0, 1}},
	     TextureStateTransition{
	         m_thumbnail.get(),
	         RHIResourceState::Undefined,
	         RHIResourceState::TransferDest,
	         TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
	    {});
	cmd_buffer->CopyBufferToTexture(staging_buffer.get(), cubemap2d.get(), 0, 0, 1);
	cmd_buffer->BlitTexture(cubemap2d.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::TransferDest,
	                        m_thumbnail.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::TransferDest);
	cmd_buffer->ResourceStateTransition(
	    {TextureStateTransition{
	         cubemap2d.get(),
	         RHIResourceState::TransferDest,
	         RHIResourceState::ShaderResource,
	         TextureRange{RHITextureDimension::Texture2D, 0, cubemap2d.get()->GetDesc().mips, 0, 1}},
	     TextureStateTransition{
	         m_thumbnail.get(),
	         RHIResourceState::TransferDest,
	         RHIResourceState::TransferSource,
	         TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}},
	     TextureStateTransition{
	         m_impl->texture.get(),
	         RHIResourceState::Undefined,
	         RHIResourceState::RenderTarget,
	         TextureRange{RHITextureDimension::Texture2DArray, 0, 1, 0, 6}}},
	    {});
	cmd_buffer->BeginRenderPass(render_target.get());
	cmd_buffer->SetViewport(512.f, 512.f);
	cmd_buffer->SetScissor(512, 512);
	cmd_buffer->BindDescriptor(descriptor);
	cmd_buffer->BindPipelineState(pipeline_state.get());
	cmd_buffer->Draw(3, 6);
	cmd_buffer->EndRenderPass();
	cmd_buffer->CopyTextureToBuffer(m_thumbnail.get(), staging_buffer.get(), 0, 0, 1);
	cmd_buffer->ResourceStateTransition(
	    {TextureStateTransition{
	        m_impl->texture.get(),
	        RHIResourceState::RenderTarget,
	        RHIResourceState::TransferSource,
	        TextureRange{RHITextureDimension::TextureCube, 0, 1, 0, 6}}},
	    {});
	cmd_buffer->CopyTextureToBuffer(m_impl->texture.get(), cubemap_buffer.get(), 0, 0, 6);
	cmd_buffer->ResourceStateTransition(
	    {TextureStateTransition{
	        m_impl->texture.get(),
	        RHIResourceState::TransferSource,
	        RHIResourceState::ShaderResource,
	        TextureRange{RHITextureDimension::TextureCube, 0, 1, 0, 6}}},
	    {});
	cmd_buffer->End();

	rhi_context->Execute(cmd_buffer);

	std::vector<uint8_t> thumbnail_data(4 * 128 * 128);
	staging_buffer->CopyToHost(thumbnail_data.data(), thumbnail_data.size());

	std::vector<uint8_t> cubemap_data(cubemap_buffer->GetDesc().size);
	cubemap_buffer->CopyToHost(cubemap_data.data(), cubemap_data.size());

	SERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::TextureCube), thumbnail_data, cubemap_desc, cubemap_data);
}

Resource<ResourceType::TextureCube>::~Resource()
{
	m_impl.reset();
}

bool Resource<ResourceType::TextureCube>::Validate() const
{
	return m_impl != nullptr;
}

void Resource<ResourceType::TextureCube>::Load(RHIContext *rhi_context)
{
	std::vector<uint8_t> thumbnail_data, cubemap_data;
	TextureDesc          desc;

	DESERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::TextureCube), thumbnail_data, desc, cubemap_data);

	m_impl = std::make_unique<Impl>();

	m_impl->texture = rhi_context->CreateTexture(desc);

	auto staging_buffer = rhi_context->CreateBuffer(cubemap_data.size(), RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU);
	staging_buffer->CopyToDevice(cubemap_data.data(), cubemap_data.size());

	auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
	cmd_buffer->Begin();
	cmd_buffer->ResourceStateTransition(
	    {TextureStateTransition{
	        m_impl->texture.get(),
	        RHIResourceState::Undefined,
	        RHIResourceState::TransferDest,
	        TextureRange{RHITextureDimension::TextureCube, 0, m_impl->texture.get()->GetDesc().mips, 0, m_impl->texture.get()->GetDesc().layers}}},
	    {});
	cmd_buffer->CopyBufferToTexture(staging_buffer.get(), m_impl->texture.get(), 0, 0, m_impl->texture.get()->GetDesc().layers);
	cmd_buffer->ResourceStateTransition(
	    {TextureStateTransition{
	        m_impl->texture.get(),
	        RHIResourceState::TransferDest,
	        RHIResourceState::ShaderResource,
	        TextureRange{RHITextureDimension::TextureCube, 0, m_impl->texture.get()->GetDesc().mips, 0, m_impl->texture.get()->GetDesc().layers}}},
	    {});
	cmd_buffer->End();

	rhi_context->Execute(cmd_buffer);
}

RHITexture *Resource<ResourceType::TextureCube>::GetTexture() const
{
	return m_impl->texture.get();
}
}        // namespace Ilum