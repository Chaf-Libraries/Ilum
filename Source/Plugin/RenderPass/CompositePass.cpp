#include "IPass.hpp"

#include <bitset>

using namespace Ilum;

class CompositePass : public RenderPass<CompositePass>
{
  public:
	CompositePass() = default;

	~CompositePass() = default;

	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::Compute)
		    .SetName("CompositePass")
		    .SetCategory("Shading")
		    .ReadTexture2D(handle++, "DI", RHIResourceState::ShaderResource)
		    .ReadTexture2D(handle++, "Environment", RHIResourceState::ShaderResource)
		    .ReadTexture2D(handle++, "AO", RHIResourceState::ShaderResource)
		    .WriteTexture2D(handle++, "Output", 0, 0, RHIFormat::R32G32B32A32_FLOAT, RHIResourceState::UnorderedAccess);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		auto *rhi_context = renderer->GetRHIContext();

		auto pipeline_state = std::shared_ptr<RHIPipelineState>(std::move(rhi_context->CreatePipelineState()));

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto direct_illumination = render_graph.GetTexture(desc.GetPin("DI").handle);
			auto environment         = render_graph.GetTexture(desc.GetPin("Environment").handle);
			auto output              = render_graph.GetTexture(desc.GetPin("Output").handle);

			auto *gpu_scene = black_board.Get<GPUScene>();
			auto *view      = black_board.Get<View>();

			auto shader = renderer->RequireShader("Source/Shaders/Shading/Composite.hlsl", "CSmain", RHIShaderStage::Compute,
			                                      {
			                                          direct_illumination ? "HAS_DIRECT_ILLUMINATION" : "NO_DIRECT_ILLUMINATION",
			                                          environment ? "HAS_ENVIRONMENT" : "NO_ENVIRONMENT",
			                                      });

			ShaderMeta shader_meta = renderer->RequireShaderMeta(shader);

			pipeline_state->ClearShader();
			pipeline_state->SetShader(RHIShaderStage::Compute, shader);

			auto descriptor = rhi_context->CreateDescriptor(shader_meta);

			descriptor->BindTexture("Output", output, RHITextureDimension::Texture2D);

			if (direct_illumination)
			{
				descriptor->BindTexture("DirectIllumination", direct_illumination, RHITextureDimension::Texture2D);
			}

			if (environment)
			{
				descriptor->BindTexture("Environment", environment, RHITextureDimension::Texture2D);
			}

			cmd_buffer->BindDescriptor(descriptor);
			cmd_buffer->BindPipelineState(pipeline_state.get());
			cmd_buffer->Dispatch(output->GetDesc().width, output->GetDesc().height, 1, 8, 8, 1);
		};
	}

	virtual void OnImGui(Variant *config)
	{
	}
};

CONFIGURATION_PASS(CompositePass)