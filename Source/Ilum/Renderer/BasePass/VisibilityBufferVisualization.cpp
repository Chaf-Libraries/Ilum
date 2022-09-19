#include "VisibilityBufferVisualization.hpp"

namespace Ilum
{
RenderPassDesc VisibilityBufferVisualization::CreateDesc()
{
	RenderPassDesc desc = {};
	desc.SetName("VisibilityBufferVisualization")
	    .SetBindPoint(BindPoint::Compute)
	    .Read("VisibilityBuffer", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
	    .Write("InstanceID", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess)
	    .Write("PrimitiveID", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess)
	    .Write("MeshletID", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess);
	return desc;
}

RenderGraph::RenderTask VisibilityBufferVisualization::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	ShaderMeta meta;

	auto *shader = renderer->RequireShader("Source/Shaders/VisibilityBufferVisualization.hlsl", "CSmain", RHIShaderStage::Compute);
	meta += renderer->RequireShaderMeta(shader);

	std::shared_ptr<RHIPipelineState> pipeline_state = std::move(renderer->GetRHIContext()->CreatePipelineState());
	std::shared_ptr<RHIDescriptor>    descriptor     = std::move(renderer->GetRHIContext()->CreateDescriptor(meta));

	pipeline_state->SetShader(RHIShaderStage::Compute, shader);

	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		auto *visibility_buffer   = render_graph.GetTexture(desc.resources.at("VisibilityBuffer").handle);
		auto *instance_id_buffer  = render_graph.GetTexture(desc.resources.at("InstanceID").handle);
		auto *primitive_id_buffer = render_graph.GetTexture(desc.resources.at("PrimitiveID").handle);
		auto *meshlet_id_buffer   = render_graph.GetTexture(desc.resources.at("MeshletID").handle);

		descriptor
		    ->BindTexture("VisibilityBuffer", visibility_buffer, RHITextureDimension::Texture2D)
		    .BindTexture("InstanceID", instance_id_buffer, RHITextureDimension::Texture2D)
		    .BindTexture("PrimitiveID", primitive_id_buffer, RHITextureDimension::Texture2D)
		    .BindTexture("MeshletID", meshlet_id_buffer, RHITextureDimension::Texture2D);

		cmd_buffer->BindDescriptor(descriptor.get());
		cmd_buffer->BindPipelineState(pipeline_state.get());
		cmd_buffer->Dispatch(visibility_buffer->GetDesc().width, visibility_buffer->GetDesc().height, 1, 32, 32, 1);
	};
}
}        // namespace Ilum