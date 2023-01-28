#include "IPass.hpp"

using namespace Ilum;

class VisibilityBufferVisualization : public RenderPass<VisibilityBufferVisualization>
{
  public:
	VisibilityBufferVisualization() = default;

	~VisibilityBufferVisualization() = default;

	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::Compute)
		    .SetName("VisibilityBufferVisualization")
		    .SetCategory("RenderPath")
		    .ReadTexture2D(handle++, "Visibility Buffer", RHIResourceState::ShaderResource)
		    .ReadTexture2D(handle++, "Depth Buffer", RHIResourceState::ShaderResource)
		    .WriteTexture2D(handle++, "Instance ID", 0, 0, RHIFormat::R8G8B8A8_UNORM, RHIResourceState::UnorderedAccess)
		    .WriteTexture2D(handle++, "Primitive ID", 0, 0, RHIFormat::R8G8B8A8_UNORM, RHIResourceState::UnorderedAccess);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		ShaderMeta meta;

		auto *shader = renderer->RequireShader("Source/Shaders/RenderPath/VisibilityBufferVisualization.hlsl", "CSmain", RHIShaderStage::Compute);
		meta += renderer->RequireShaderMeta(shader);

		std::shared_ptr<RHIPipelineState> pipeline_state = std::move(renderer->GetRHIContext()->CreatePipelineState());

		pipeline_state->SetShader(RHIShaderStage::Compute, shader);

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto *visibility_buffer   = render_graph.GetTexture(desc.GetPin("Visibility Buffer").handle);
			auto *depth_buffer        = render_graph.GetTexture(desc.GetPin("Depth Buffer").handle);
			auto *instance_id_buffer  = render_graph.GetTexture(desc.GetPin("Instance ID").handle);
			auto *primitive_id_buffer = render_graph.GetTexture(desc.GetPin("Primitive ID").handle);

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
