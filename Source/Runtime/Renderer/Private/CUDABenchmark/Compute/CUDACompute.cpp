#include "CUDACompute.hpp"

namespace Ilum
{
RenderPassDesc CUDACompute::CreateDesc()
{
	RenderPassDesc desc;

	return desc.SetName<CUDACompute>()
	    .SetBindPoint(BindPoint::CUDA)
	    .Write("Result", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess);
}

RenderGraph::RenderTask CUDACompute::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	auto *shader = renderer->RequireShader("Source/Shaders/CUDATest/Compute.hlsl", "MainCS", RHIShaderStage::Compute, {}, {}, true);

	ShaderMeta meta = renderer->RequireShaderMeta(shader);

	std::shared_ptr<RHIDescriptor>    descriptor     = std::move(renderer->GetRHIContext()->CreateDescriptor(meta, true));
	std::shared_ptr<RHIPipelineState> pipeline_state = std::move(renderer->GetRHIContext()->CreatePipelineState(true));

	pipeline_state->SetShader(RHIShaderStage::Compute, shader);

	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		auto *result = render_graph.GetCUDATexture(desc.resources.at("Result").handle);

		descriptor
		    ->BindTexture("Result", result, RHITextureDimension::Texture2D);

		cmd_buffer->BindDescriptor(descriptor.get());
		cmd_buffer->BindPipelineState(pipeline_state.get());
		cmd_buffer->Dispatch(result->GetDesc().width, result->GetDesc().height, 1, 8, 8, 1);
	};
}
}        // namespace Ilum