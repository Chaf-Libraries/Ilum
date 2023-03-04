#include "IPass.hpp"

#include <bitset>

using namespace Ilum;

class FXAA : public RenderPass<FXAA>
{
	enum class Quality
	{
		Low,
		Medium,
		High
	};

	struct Config
	{
		float   fixed_threshold    = 0.8333f;
		float   relative_threshold = 0.166f;
		float   subpixel_blending  = 0.7f;
		Quality quality            = Quality::High;
	};

  public:
	FXAA() = default;

	~FXAA() = default;

	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::Compute)
		    .SetName("FXAA")
		    .SetCategory("PostProcess")
		    .SetConfig(Config())
		    .ReadTexture2D(handle++, "Input", RHIResourceState::ShaderResource)
		    .WriteTexture2D(handle++, "Output", RHIFormat::R32G32B32A32_FLOAT, RHIResourceState::UnorderedAccess);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		const char *fxaa_qualities[] = {"FXAA_QUALITY_LOW", "FXAA_QUALITY_MEDIUM", "FXAA_QUALITY_HIGH"};

		auto *rhi_context = renderer->GetRHIContext();

		auto pipeline_state = std::shared_ptr<RHIPipelineState>(std::move(rhi_context->CreatePipelineState()));

		std::shared_ptr<RHIBuffer> config_buffer = rhi_context->CreateBuffer<Config>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto input  = render_graph.GetTexture(desc.GetPin("Input").handle);
			auto output = render_graph.GetTexture(desc.GetPin("Output").handle);

			auto *gpu_scene = black_board.Get<GPUScene>();
			auto *view      = black_board.Get<View>();

			config_buffer->CopyToDevice(config.Convert<Config>(), sizeof(Config));

			auto       shader      = renderer->RequireShader("Source/Shaders/PostProcess/FXAA.hlsl", "CSmain", RHIShaderStage::Compute, {fxaa_qualities[static_cast<int32_t>(config.Convert<Config>()->quality)]});
			ShaderMeta shader_meta = renderer->RequireShaderMeta(shader);
			auto       descriptor  = rhi_context->CreateDescriptor(shader_meta);

			pipeline_state->ClearShader();
			pipeline_state->SetShader(RHIShaderStage::Compute, shader);

			descriptor->BindTexture("Input", input, RHITextureDimension::Texture2D)
			    .BindTexture("Output", output, RHITextureDimension::Texture2D)
			    .BindSampler("Sampler", rhi_context->CreateSampler(SamplerDesc::LinearClamp()))
			    .BindBuffer("ConfigBuffer", config_buffer.get());

			cmd_buffer->BindDescriptor(descriptor);
			cmd_buffer->BindPipelineState(pipeline_state.get());
			cmd_buffer->Dispatch(output->GetDesc().width, output->GetDesc().height, 1, 8, 8, 1);
		};
	}

	virtual void OnImGui(Variant *config)
	{
		auto *config_data = config->Convert<Config>();

		const char *qualities[] = {"Low", "Medium", "High"};

		ImGui::Combo("Quality", reinterpret_cast<int32_t *>(&config_data->quality), qualities, 3);
		ImGui::SliderFloat("Fixed Threshold", &config_data->fixed_threshold, 0.0312f, 0.0833f, "%.4f");
		ImGui::SliderFloat("Relative Threshold", &config_data->relative_threshold, 0.063f, 0.333f, "%.4f");
		ImGui::SliderFloat("Subpixel Blending", &config_data->subpixel_blending, 0.f, 1.f, "%.2f");
	}
};

CONFIGURATION_PASS(FXAA)