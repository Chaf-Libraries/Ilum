#include "IPass.hpp"

using namespace Ilum;

class VisibilityBufferVisualization : public IPass<VisibilityBufferVisualization>
{
  public:
	VisibilityBufferVisualization() = default;

	~VisibilityBufferVisualization() = default;

	virtual void CreateDesc(RenderPassDesc *desc)
	{
		desc->SetBindPoint(BindPoint::Compute)
		    .Read("VisibilityBuffer", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
		    .Read("DepthBuffer", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
		    .Write("InstanceID", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess)
		    .Write("PrimitiveID", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		ShaderMeta meta;

		auto *shader = renderer->RequireShader("Source/Shaders/VisibilityBufferVisualization.hlsl", "CSmain", RHIShaderStage::Compute);
		meta += renderer->RequireShaderMeta(shader);

		std::shared_ptr<RHIPipelineState> pipeline_state = std::move(renderer->GetRHIContext()->CreatePipelineState());

		pipeline_state->SetShader(RHIShaderStage::Compute, shader);

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto *visibility_buffer   = render_graph.GetTexture(desc.resources.at("VisibilityBuffer").handle);
			auto *depth_buffer        = render_graph.GetTexture(desc.resources.at("DepthBuffer").handle);
			auto *instance_id_buffer  = render_graph.GetTexture(desc.resources.at("InstanceID").handle);
			auto *primitive_id_buffer = render_graph.GetTexture(desc.resources.at("PrimitiveID").handle);

			auto *descriptor = renderer->GetRHIContext()->CreateDescriptor(meta);

			descriptor->BindTexture("VisibilityBuffer", visibility_buffer, RHITextureDimension::Texture2D)
			    .BindTexture("DepthBuffer", depth_buffer, RHITextureDimension::Texture2D)
			    .BindTexture("InstanceID", instance_id_buffer, RHITextureDimension::Texture2D)
			    .BindTexture("PrimitiveID", primitive_id_buffer, RHITextureDimension::Texture2D);

			cmd_buffer->BindDescriptor(descriptor);
			cmd_buffer->BindPipelineState(pipeline_state.get());
			cmd_buffer->Dispatch(visibility_buffer->GetDesc().width, visibility_buffer->GetDesc().height, 1, 32, 32, 1);
		};
	}

	virtual void OnImGui(Variant *config)
	{
	}
};

CONFIGURATION_PASS(VisibilityBufferVisualization)
