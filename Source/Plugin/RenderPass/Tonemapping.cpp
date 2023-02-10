#include "IPass.hpp"

#include <bitset>

using namespace Ilum;

class Tonemapping : public RenderPass<Tonemapping>
{
	struct Config
	{
		float   brightness    = 1.f;
		float   contrast      = 1.f;
		float   saturation    = 1.f;
		float   vignette      = 0.f;
		float   avg_lum       = 1.f;
		int32_t auto_exposure = 0;
		float   Ywhite        = 0.5f;        // Burning white
		float   key           = 0.5f;        // Log-average luminance
	};

  public:
	Tonemapping() = default;

	~Tonemapping() = default;

	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::Compute)
		    .SetName("Tonemapping")
		    .SetCategory("PostProcess")
		    .SetConfig(Config())
		    .ReadTexture2D(handle++, "Input", RHIResourceState::ShaderResource)
		    .WriteTexture2D(handle++, "Output", 0, 0, RHIFormat::R16G16B16A16_FLOAT, RHIResourceState::UnorderedAccess);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		auto *rhi_context    = renderer->GetRHIContext();
		auto  shader      = renderer->RequireShader("Source/Shaders/PostProcess/Tonemapping.hlsl", "CSmain", RHIShaderStage::Compute);
		ShaderMeta shader_meta    = renderer->RequireShaderMeta(shader);

		auto  pipeline_state = std::shared_ptr<RHIPipelineState>(std::move(rhi_context->CreatePipelineState()));
		pipeline_state->SetShader(RHIShaderStage::Compute, shader);

		std::shared_ptr<RHIBuffer> uniform_buffer = rhi_context->CreateBuffer<Config>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto input  = render_graph.GetTexture(desc.GetPin("Input").handle);
			auto output = render_graph.GetTexture(desc.GetPin("Output").handle);

			auto *gpu_scene = black_board.Get<GPUScene>();
			auto *view      = black_board.Get<View>();

			uniform_buffer->CopyToDevice(config.Convert<Config>(), sizeof(Config));

			auto descriptor = rhi_context->CreateDescriptor(shader_meta);

			descriptor->BindTexture("Input", input, RHITextureDimension::Texture2D)
			    .BindTexture("Output", output, RHITextureDimension::Texture2D)
			    .BindSampler("TexSampler", rhi_context->CreateSampler(SamplerDesc::NearestClamp()))
				.BindBuffer("UniformBuffer", uniform_buffer.get());

			cmd_buffer->BindDescriptor(descriptor);
			cmd_buffer->BindPipelineState(pipeline_state.get());
			cmd_buffer->Dispatch(output->GetDesc().width, output->GetDesc().height, 1, 8, 8, 1);
		};
	}

	virtual void OnImGui(Variant *config)
	{
		auto *config_data = config->Convert<Config>();

		std::bitset<8> b(config_data->auto_exposure);

		bool auto_exposure = b.test(0);

		ImGui::Checkbox("Auto Exposure", &auto_exposure);
		ImGui::SliderFloat("Exposure", &config_data->avg_lum, 0.001f, 5.0f, "%.3f");
		ImGui::SliderFloat("Brightness", &config_data->brightness, 0.0f, 2.0f, "%.3f");
		ImGui::SliderFloat("Contrast", &config_data->contrast, 0.0f, 2.0f, "%.3f");
		ImGui::SliderFloat("Saturation", &config_data->saturation, 0.0f, 5.0f, "%.3f");
		ImGui::SliderFloat("Vignette", &config_data->vignette, 0.0f, 2.0f, "%.3f");

		if (auto_exposure)
		{
			bool localExposure = b.test(1);
			if (ImGui::TreeNode("Auto Settings"))
			{
				ImGui::Checkbox("Local", &localExposure);
				ImGui::SliderFloat("Burning White", &config_data->Ywhite, 0.f, 1.f, "%.3f");
				ImGui::SliderFloat("Brightness", &config_data->key, 0.f, 1.f, "%.3f");
				b.set(1, localExposure);
				ImGui::End();
			}
		}

		b.set(0, auto_exposure);

		config_data->auto_exposure = b.to_ulong();
	}
};

CONFIGURATION_PASS(Tonemapping)