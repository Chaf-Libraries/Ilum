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
		    .ReadTexture2D(handle++, "Env DI", RHIResourceState::ShaderResource)
		    .ReadTexture2D(handle++, "Light DI", RHIResourceState::ShaderResource)
		    .ReadTexture2D(handle++, "Environment", RHIResourceState::ShaderResource)
		    .ReadTexture2D(handle++, "AO", RHIResourceState::ShaderResource)
		    .WriteTexture2D(handle++, "Output", RHIFormat::R32G32B32A32_FLOAT, RHIResourceState::UnorderedAccess);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		auto *rhi_context = renderer->GetRHIContext();

		auto pipeline_state = std::shared_ptr<RHIPipelineState>(std::move(rhi_context->CreatePipelineState()));

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto light_direct_illumination = render_graph.GetTexture(desc.GetPin("Light DI").handle);
			auto env_direct_illumination   = render_graph.GetTexture(desc.GetPin("Env DI").handle);
			auto environment               = render_graph.GetTexture(desc.GetPin("Environment").handle);
			auto ao                        = render_graph.GetTexture(desc.GetPin("AO").handle);
			auto output                    = render_graph.GetTexture(desc.GetPin("Output").handle);

			auto *gpu_scene = black_board.Get<GPUScene>();
			auto *view      = black_board.Get<View>();

			auto shader = renderer->RequireShader("Source/Shaders/Shading/Composite.hlsl", "CSmain", RHIShaderStage::Compute,
			                                      {
			                                          light_direct_illumination ? "HAS_LIGHT_DIRECT_ILLUMINATION" : "NO_LIGHT_DIRECT_ILLUMINATION",
			                                          env_direct_illumination ? "HAS_ENV_DIRECT_ILLUMINATION" : "NO_ENV_DIRECT_ILLUMINATION",
			                                          ao ? "HAS_AMBIENT_OCCLUSION" : "NO_AMBIENT_OCCLUSION",
			                                          environment ? "HAS_ENVIRONMENT" : "NO_ENVIRONMENT",
			                                      });

			ShaderMeta shader_meta = renderer->RequireShaderMeta(shader);

			pipeline_state->ClearShader();
			pipeline_state->SetShader(RHIShaderStage::Compute, shader);

			auto descriptor = rhi_context->CreateDescriptor(shader_meta);

			descriptor->BindTexture("Output", output, RHITextureDimension::Texture2D);

			if (light_direct_illumination)
			{
				descriptor->BindTexture("LightDirectIllumination", light_direct_illumination, RHITextureDimension::Texture2D);
			}

			if (env_direct_illumination)
			{
				descriptor->BindTexture("EnvDirectIllumination", env_direct_illumination, RHITextureDimension::Texture2D);
			}

			if (environment)
			{
				descriptor->BindTexture("Environment", environment, RHITextureDimension::Texture2D);
			}

			if (ao)
			{
				descriptor->BindTexture("AmbientOcclusion", ao, RHITextureDimension::Texture2D);
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