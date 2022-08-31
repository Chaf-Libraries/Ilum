#include "Renderer.hpp"

#include <Core/Path.hpp>

#include <RenderCore/ShaderCompiler/ShaderCompiler.hpp>

//#include <RHI/Backend/CUDA/Device.hpp>

namespace Ilum
{
Renderer::Renderer(RHIContext *rhi_context):
    p_rhi_context(rhi_context)
{
	/*m_texture = p_rhi_context->CreateTexture2D(100, 100, RHIFormat::R32G32B32A32_FLOAT, RHITextureUsage::UnorderedAccess | RHITextureUsage::ShaderResource, false);
	m_buffer  = p_rhi_context->CreateBuffer(sizeof(glm::uvec2), RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);

	glm::uvec2 tex_size = {100, 100};
	std::memcpy(m_buffer->Map(), &tex_size, sizeof(tex_size));
	m_buffer->Unmap();

	std::vector<uint8_t> data;
	Path::GetInstance().Read("Source/Shaders/DrawUV.hlsl", data);
	ShaderDesc desc = {};
	desc.code.resize(data.size());
	std::memcpy(desc.code.data(), data.data(), data.size());
	desc.entry_point = "CSmain";
	desc.source      = ShaderSource::HLSL;
	desc.target      = ShaderTarget::SPIRV;
	desc.stage       = RHIShaderStage::Compute;

	ShaderMeta meta;
	m_shader         = p_rhi_context->CreateShader("CSmain", ShaderCompiler::GetInstance().Compile(desc, meta));
	m_descriptor     = p_rhi_context->CreateDescriptor(meta);
	m_pipeline_state = p_rhi_context->CreatePipelineState();
	m_pipeline_state->SetShader(RHIShaderStage::Compute, m_shader.get());

	m_descriptor->BindTexture("Result", m_texture.get(), RHITextureDimension::Texture2D);
	m_descriptor->BindBuffer("TexSize", m_buffer.get());

	auto *cmd_buffer = p_rhi_context->CreateCommand(RHIQueueFamily::Graphics);
	cmd_buffer->Begin();
	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	                                        m_texture.get(),
	                                        RHIResourceState::Undefined,
	                                        RHIResourceState::ShaderResource,
	                                        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
	                                    {});
	cmd_buffer->End();
	p_rhi_context->GetQueue(RHIQueueFamily::Graphics)->Submit({cmd_buffer});*/
}

Renderer::~Renderer()
{
}

void Renderer::Tick()
{
	/*auto *cmd_buffer = p_rhi_context->CreateCommand(RHIQueueFamily::Graphics);
	cmd_buffer->Begin();
	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	                                        m_texture.get(),
	                                        RHIResourceState::ShaderResource,
	                                        RHIResourceState::UnorderedAccess,
	                                        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
	                                    {});
	cmd_buffer->BindDescriptor(m_descriptor.get());
	cmd_buffer->BindPipelineState(m_pipeline_state.get());
	cmd_buffer->Dispatch(100, 100, 1, 8, 8, 1);
	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	                                        m_texture.get(),
	                                        RHIResourceState::UnorderedAccess,
	                                        RHIResourceState::ShaderResource,
	                                        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
	                                    {});
	cmd_buffer->End();
	p_rhi_context->GetQueue(RHIQueueFamily::Graphics)->Submit({cmd_buffer});*/
}

void Renderer::SetRenderGraph(std::unique_ptr<RenderGraph> &&render_graph)
{
	m_render_graph = std::move(render_graph);
}

RenderGraph *Renderer::GetRenderGraph() const
{
	return m_render_graph.get();
}

RHITexture *Renderer::GetTexture()
{
	return m_texture.get();
}

}        // namespace Ilum