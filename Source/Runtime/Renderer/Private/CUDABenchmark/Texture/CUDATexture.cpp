#include "CUDATexture.hpp"

#include <Resource/ResourceManager.hpp>

namespace Ilum
{
RenderPassDesc CUDATexture::CreateDesc()
{
	RenderPassDesc desc;

	return desc.SetName<CUDATexture>()
	    .SetBindPoint(BindPoint::CUDA)
	    .Write("Result", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess);
}

RenderGraph::RenderTask CUDATexture::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	auto *shader = renderer->RequireShader("Source/Shaders/CUDATest/Texture.hlsl", "MainCS", RHIShaderStage::Compute, {}, {}, true);

	ShaderMeta meta = renderer->RequireShaderMeta(shader);

	std::shared_ptr<RHIDescriptor>    descriptor     = std::move(renderer->GetRHIContext()->CreateDescriptor(meta, true));
	std::shared_ptr<RHIPipelineState> pipeline_state = std::move(renderer->GetRHIContext()->CreatePipelineState(true));

	size_t tex_id = renderer->GetResourceManager()->Import<ResourceType::Texture>("Asset/Texture/Default/default.png");

	auto texture =renderer->GetResourceManager()->GetResource<ResourceType::Texture>(tex_id)->GetTexture();

	std::shared_ptr<RHITexture> gfx_texture  = std::move(renderer->GetRHIContext()->CreateTexture2D(
	    texture->GetDesc().width,
	    texture->GetDesc().height,
	     RHIFormat::R32G32B32A32_FLOAT, RHITextureUsage::Transfer, false, 1));
	{
		auto *cmd_buffer = renderer->GetRHIContext()->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->BlitTexture(texture, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::ShaderResource, gfx_texture.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::Undefined, RHIFilter::Nearest);
		cmd_buffer->End();
		renderer->GetRHIContext()->Execute(cmd_buffer);
	}
	std::shared_ptr<RHITexture> cuda_texture = renderer->GetRHIContext()->MapToCUDATexture(gfx_texture.get());

	pipeline_state->SetShader(RHIShaderStage::Compute, shader);

	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		auto *result = render_graph.GetCUDATexture(desc.resources.at("Result").handle);

		descriptor
		    ->BindTexture("Texture", cuda_texture.get(), RHITextureDimension::Texture2D)
		    .BindSampler("Sampler", renderer->GetRHIContext()->CreateSampler(SamplerDesc::LinearClamp))
		    .BindTexture("Result", result, RHITextureDimension::Texture2D);

		cmd_buffer->BindDescriptor(descriptor.get());
		cmd_buffer->BindPipelineState(pipeline_state.get());
		cmd_buffer->Dispatch(result->GetDesc().width, result->GetDesc().height, 1, 8, 8, 1);
	};
}
}        // namespace Ilum