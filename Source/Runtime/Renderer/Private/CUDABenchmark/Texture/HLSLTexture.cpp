#include "HLSLTexture.hpp"

#include <Resource/ResourceManager.hpp>

namespace Ilum
{
RenderPassDesc HLSLTexture::CreateDesc()
{
	RenderPassDesc desc;

	return desc.SetName<HLSLTexture>()
	    .SetBindPoint(BindPoint::Compute)
	    .Write("Result", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess);
}

RenderGraph::RenderTask HLSLTexture::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	auto *shader = renderer->RequireShader("Source/Shaders/CUDATest/Texture.hlsl", "MainCS", RHIShaderStage::Compute, {}, {});

	ShaderMeta meta = renderer->RequireShaderMeta(shader);

	std::shared_ptr<RHIDescriptor>    descriptor     = std::move(renderer->GetRHIContext()->CreateDescriptor(meta));
	std::shared_ptr<RHIPipelineState> pipeline_state = std::move(renderer->GetRHIContext()->CreatePipelineState());

	size_t tex_id = renderer->GetResourceManager()->Import<ResourceType::Texture>("Asset/Texture/Default/default.png");

	auto *texture = renderer->GetResourceManager()->GetResource<ResourceType::Texture>(tex_id);

	pipeline_state->SetShader(RHIShaderStage::Compute, shader);

	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		auto *result = render_graph.GetTexture(desc.resources.at("Result").handle);

		descriptor
		    ->BindTexture("Texture", texture->GetTexture(), RHITextureDimension::Texture2D)
		    .BindSampler("Sampler", renderer->GetRHIContext()->CreateSampler(SamplerDesc::LinearClamp))
		    .BindTexture("Result", result, RHITextureDimension::Texture2D);

		cmd_buffer->BindDescriptor(descriptor.get());
		cmd_buffer->BindPipelineState(pipeline_state.get());
		cmd_buffer->Dispatch(result->GetDesc().width, result->GetDesc().height, 1, 8, 8, 1);
	};
}
}        // namespace Ilum