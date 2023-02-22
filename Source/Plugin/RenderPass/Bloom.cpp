#include "IPass.hpp"

#include <bitset>

using namespace Ilum;

class Bloom : public RenderPass<Bloom>
{
	struct Config
	{
		float threshold = 0.75f;
		float radius    = 0.75f;
		float intensity = 1.f;
	};

	struct PipelineDesc
	{
		std::shared_ptr<RHIPipelineState> pipeline_state = nullptr;
		ShaderMeta                        shader_meta;
	};

	struct BloomPassData
	{
		std::unique_ptr<RHITexture> mask     = nullptr;
		std::unique_ptr<RHITexture> level[4] = {nullptr};
		std::unique_ptr<RHITexture> blur[4]  = {nullptr};
	};

  public:
	Bloom() = default;

	~Bloom() = default;

	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::Compute)
		    .SetName("Bloom")
		    .SetCategory("PostProcess")
		    .SetConfig(Config())
		    .ReadTexture2D(handle++, "Input", RHIResourceState::ShaderResource)
		    .WriteTexture2D(handle++, "Output", 0, 0, RHIFormat::R32G32B32A32_FLOAT, RHIResourceState::UnorderedAccess);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		auto *rhi_context = renderer->GetRHIContext();

		auto bloom_mask          = CreatePipelineDesc(renderer, "BloomMask");
		auto bloom_down_sampling = CreatePipelineDesc(renderer, "BloomDownSampling");
		auto bloom_blur          = CreatePipelineDesc(renderer, "BloomBlur");
		auto bloom_up_sampling   = CreatePipelineDesc(renderer, "BloomUpSampling");
		auto bloom_blend         = CreatePipelineDesc(renderer, "BloomBlend");

		std::shared_ptr<RHIBuffer> uniform_buffer = rhi_context->CreateBuffer<Config>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto input  = render_graph.GetTexture(desc.GetPin("Input").handle);
			auto output = render_graph.GetTexture(desc.GetPin("Output").handle);

			auto *gpu_scene  = black_board.Get<GPUScene>();
			auto *view       = black_board.Get<View>();
			auto *bloom_data = black_board.Get<BloomPassData>();

			if (!bloom_data->mask ||
			    (bloom_data->mask->GetDesc().width != input->GetDesc().width ||
			     bloom_data->mask->GetDesc().height != input->GetDesc().height))
			{
				bloom_data->mask = rhi_context->CreateTexture2D(input->GetDesc().width, input->GetDesc().height, RHIFormat::R32G32B32A32_FLOAT, RHITextureUsage::UnorderedAccess | RHITextureUsage::ShaderResource, false);
				cmd_buffer->ResourceStateTransition({TextureStateTransition{bloom_data->mask.get(), RHIResourceState::Undefined, RHIResourceState::UnorderedAccess}}, {});
			}

			if (!bloom_data->level[0] ||
			    (bloom_data->level[0]->GetDesc().width != (input->GetDesc().width >> 1) ||
			     bloom_data->level[0]->GetDesc().height != (input->GetDesc().height >> 1)))
			{
				for (uint32_t i = 0; i < 4; i++)
				{
					bloom_data->level[i] = rhi_context->CreateTexture2D(input->GetDesc().width >> (i + 1), input->GetDesc().height >> (i + 1), RHIFormat::R32G32B32A32_FLOAT, RHITextureUsage::UnorderedAccess | RHITextureUsage::ShaderResource, false);
				}
				cmd_buffer->ResourceStateTransition(
				    {TextureStateTransition{bloom_data->level[0].get(), RHIResourceState::Undefined, RHIResourceState::UnorderedAccess},
				     TextureStateTransition{bloom_data->level[1].get(), RHIResourceState::Undefined, RHIResourceState::UnorderedAccess},
				     TextureStateTransition{bloom_data->level[2].get(), RHIResourceState::Undefined, RHIResourceState::UnorderedAccess},
				     TextureStateTransition{bloom_data->level[3].get(), RHIResourceState::Undefined, RHIResourceState::UnorderedAccess}},
				    {});
			}

			if (!bloom_data->blur[0] ||
			    (bloom_data->blur[0]->GetDesc().width != (input->GetDesc().width >> 1) ||
			     bloom_data->blur[0]->GetDesc().height != (input->GetDesc().height >> 1)))
			{
				for (uint32_t i = 0; i < 4; i++)
				{
					bloom_data->blur[i] = rhi_context->CreateTexture2D(input->GetDesc().width >> (i + 1), input->GetDesc().height >> (i + 1), RHIFormat::R32G32B32A32_FLOAT, RHITextureUsage::UnorderedAccess | RHITextureUsage::ShaderResource, false);
				}
				cmd_buffer->ResourceStateTransition(
				    {TextureStateTransition{bloom_data->blur[0].get(), RHIResourceState::Undefined, RHIResourceState::UnorderedAccess},
				     TextureStateTransition{bloom_data->blur[1].get(), RHIResourceState::Undefined, RHIResourceState::UnorderedAccess},
				     TextureStateTransition{bloom_data->blur[2].get(), RHIResourceState::Undefined, RHIResourceState::UnorderedAccess},
				     TextureStateTransition{bloom_data->blur[3].get(), RHIResourceState::Undefined, RHIResourceState::UnorderedAccess}},
				    {});
			}

			uniform_buffer->CopyToDevice(config.Convert<Config>(), sizeof(Config));

			// Masking
			{
				cmd_buffer->BeginMarker("Bloom Mask");
				auto descriptor = rhi_context->CreateDescriptor(bloom_mask.shader_meta);
				descriptor->BindTexture("BloomMaskInput", input, RHITextureDimension::Texture2D)
				    .BindTexture("BloomMaskOutput", bloom_data->mask.get(), RHITextureDimension::Texture2D)
				    .BindBuffer("ConfigBuffer", uniform_buffer.get());
				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(bloom_mask.pipeline_state.get());
				cmd_buffer->Dispatch(input->GetDesc().width, input->GetDesc().height, 1, 8, 8, 1);
				cmd_buffer->EndMarker();
			}

			// Resource Transition
			{
				cmd_buffer->ResourceStateTransition({TextureStateTransition{bloom_data->mask.get(), RHIResourceState::UnorderedAccess, RHIResourceState::ShaderResource}}, {});
			}

			// Down Sampling
			{
				cmd_buffer->BeginMarker("Bloom Down Sampling");
				for (uint32_t i = 0; i < 4; i++)
				{
					cmd_buffer->BeginMarker(fmt::format("Bloom Down Sampling - {}", i));
					auto descriptor = rhi_context->CreateDescriptor(bloom_down_sampling.shader_meta);
					descriptor->BindTexture("BloomDownSamplingInput", i == 0 ? bloom_data->mask.get() : bloom_data->level[i - 1].get(), RHITextureDimension::Texture2D)
					    .BindTexture("BloomDownSamplingOutput", bloom_data->level[i].get(), RHITextureDimension::Texture2D)
					    .BindSampler("BloomDownSampleSampler", rhi_context->CreateSampler(SamplerDesc::LinearClamp()));
					cmd_buffer->BindDescriptor(descriptor);
					cmd_buffer->BindPipelineState(bloom_down_sampling.pipeline_state.get());
					cmd_buffer->Dispatch(bloom_data->level[i]->GetDesc().width, bloom_data->level[i]->GetDesc().height, 1, 8, 8, 1);
					cmd_buffer->ResourceStateTransition({TextureStateTransition{bloom_data->level[i].get(), RHIResourceState::UnorderedAccess, RHIResourceState::ShaderResource}}, {});
					cmd_buffer->EndMarker();
				}
				cmd_buffer->EndMarker();
			}

			// Blurring
			{
				cmd_buffer->BeginMarker("Bloom Blur");
				for (uint32_t i = 0; i < 4; i++)
				{
					cmd_buffer->BeginMarker(fmt::format("Bloom Blur - {}", i));
					auto descriptor = rhi_context->CreateDescriptor(bloom_blur.shader_meta);
					descriptor->BindTexture("BloomBlurInput", bloom_data->level[i].get(), RHITextureDimension::Texture2D)
					    .BindTexture("BloomBlurOutput", bloom_data->blur[i].get(), RHITextureDimension::Texture2D);
					cmd_buffer->BindDescriptor(descriptor);
					cmd_buffer->BindPipelineState(bloom_blur.pipeline_state.get());
					cmd_buffer->Dispatch(bloom_data->blur[i]->GetDesc().width, bloom_data->blur[i]->GetDesc().height, 1, 8, 8, 1);
					cmd_buffer->EndMarker();
				}
				cmd_buffer->EndMarker();
			}

			// Resource Transition
			{
				cmd_buffer->ResourceStateTransition(
				    {
				        TextureStateTransition{bloom_data->blur[0].get(), RHIResourceState::UnorderedAccess, RHIResourceState::ShaderResource},
				        TextureStateTransition{bloom_data->blur[1].get(), RHIResourceState::UnorderedAccess, RHIResourceState::ShaderResource},
				        TextureStateTransition{bloom_data->blur[2].get(), RHIResourceState::UnorderedAccess, RHIResourceState::ShaderResource},
				        TextureStateTransition{bloom_data->blur[3].get(), RHIResourceState::UnorderedAccess, RHIResourceState::ShaderResource},
				        TextureStateTransition{bloom_data->level[0].get(), RHIResourceState::ShaderResource, RHIResourceState::UnorderedAccess},
				        TextureStateTransition{bloom_data->level[1].get(), RHIResourceState::ShaderResource, RHIResourceState::UnorderedAccess},
				        TextureStateTransition{bloom_data->level[2].get(), RHIResourceState::ShaderResource, RHIResourceState::UnorderedAccess},
				        TextureStateTransition{bloom_data->level[3].get(), RHIResourceState::ShaderResource, RHIResourceState::UnorderedAccess},
				    },
				    {});
			}

			// Up Sampling
			{
				cmd_buffer->BeginMarker("Bloom Up Sampling");
				for (uint32_t i = 3; i >= 1; i--)
				{
					cmd_buffer->BeginMarker(fmt::format("Bloom Up Sampling - {}", i));
					auto descriptor = rhi_context->CreateDescriptor(bloom_up_sampling.shader_meta);
					descriptor->BindTexture("BloomUpSamplingLow", i == 3 ? bloom_data->blur[i].get() : bloom_data->level[i].get(), RHITextureDimension::Texture2D)
					    .BindSampler("BloomUpSamplingSampler", rhi_context->CreateSampler(SamplerDesc::LinearClamp()))
					    .BindTexture("BloomUpSamplingHigh", bloom_data->blur[i - 1].get(), RHITextureDimension::Texture2D)
					    .BindTexture("BloomUpSamplingOutput", bloom_data->level[i - 1].get(), RHITextureDimension::Texture2D)
					    .BindBuffer("ConfigBuffer", uniform_buffer.get());
					cmd_buffer->BindDescriptor(descriptor);
					cmd_buffer->BindPipelineState(bloom_up_sampling.pipeline_state.get());
					cmd_buffer->Dispatch(bloom_data->level[i - 1]->GetDesc().width, bloom_data->level[i - 1]->GetDesc().height, 1, 8, 8, 1);
					cmd_buffer->ResourceStateTransition({TextureStateTransition{bloom_data->level[i - 1].get(), RHIResourceState::UnorderedAccess, RHIResourceState::ShaderResource}}, {});
					cmd_buffer->EndMarker();
				}
				cmd_buffer->EndMarker();
			}

			// Blend
			{
				cmd_buffer->BeginMarker("Bloom Blend");
				auto descriptor = rhi_context->CreateDescriptor(bloom_blend.shader_meta);
				descriptor->BindTexture("BloomBlendBloom", bloom_data->level[0].get(), RHITextureDimension::Texture2D)
				    .BindSampler("BloomBlendSampler", rhi_context->CreateSampler(SamplerDesc::LinearClamp()))
				    .BindTexture("BloomBlendInput", input, RHITextureDimension::Texture2D)
				    .BindTexture("BloomBlendOutput", output, RHITextureDimension::Texture2D)
				    .BindBuffer("ConfigBuffer", uniform_buffer.get());
				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(bloom_blend.pipeline_state.get());
				cmd_buffer->Dispatch(output->GetDesc().width, output->GetDesc().height, 1, 8, 8, 1);
				cmd_buffer->EndMarker();
			}

			// Resource Transition
			{
				cmd_buffer->ResourceStateTransition(
				    {
				        TextureStateTransition{bloom_data->level[0].get(), RHIResourceState::ShaderResource, RHIResourceState::UnorderedAccess},
				        TextureStateTransition{bloom_data->level[1].get(), RHIResourceState::ShaderResource, RHIResourceState::UnorderedAccess},
				        TextureStateTransition{bloom_data->level[2].get(), RHIResourceState::ShaderResource, RHIResourceState::UnorderedAccess},
				        TextureStateTransition{bloom_data->blur[0].get(), RHIResourceState::ShaderResource, RHIResourceState::UnorderedAccess},
				        TextureStateTransition{bloom_data->blur[1].get(), RHIResourceState::ShaderResource, RHIResourceState::UnorderedAccess},
				        TextureStateTransition{bloom_data->blur[2].get(), RHIResourceState::ShaderResource, RHIResourceState::UnorderedAccess},
				        TextureStateTransition{bloom_data->blur[3].get(), RHIResourceState::ShaderResource, RHIResourceState::UnorderedAccess},
				        TextureStateTransition{bloom_data->mask.get(), RHIResourceState::ShaderResource, RHIResourceState::UnorderedAccess},
				    },
				    {});
			}
		};
	}

	virtual void OnImGui(Variant *config)
	{
		auto *config_data = config->Convert<Config>();

		ImGui::DragFloat("Threshold", &config_data->threshold, 0.01f, 0.f, std::numeric_limits<float>::max());
		ImGui::DragFloat("Radius", &config_data->radius, 0.01f, 0.f, std::numeric_limits<float>::max());
		ImGui::DragFloat("Intensity", &config_data->intensity, 0.01f, 0.f, std::numeric_limits<float>::max());
	}

	PipelineDesc CreatePipelineDesc(Renderer *renderer, const std::string &entry_point)
	{
		PipelineDesc pipeline_desc;

		auto shader               = renderer->RequireShader("Source/Shaders/PostProcess/Bloom.hlsl", entry_point, RHIShaderStage::Compute);
		pipeline_desc.shader_meta = renderer->RequireShaderMeta(shader);

		pipeline_desc.pipeline_state = std::shared_ptr<RHIPipelineState>(std::move(renderer->GetRHIContext()->CreatePipelineState()));
		pipeline_desc.pipeline_state->SetShader(RHIShaderStage::Compute, shader);

		return pipeline_desc;
	}
};

CONFIGURATION_PASS(Bloom)