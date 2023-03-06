#include "IPass.hpp"

#include <random>

using namespace Ilum;

#define SSAO_NOISE_DIM 4
#define SSAO_KERNEL_SIZE 64

class SSAO : public RenderPass<SSAO>
{
  public:
	SSAO() = default;

	~SSAO() = default;

	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::Compute)
		    .SetName("SSAO")
		    .SetCategory("AO")
		    .ReadTexture2D(handle++, "PositionDepth", RHIResourceState::ShaderResource)
		    .ReadTexture2D(handle++, "Normal", RHIResourceState::ShaderResource)
		    .WriteTexture2D(handle++, "Output", RHIFormat::R32_FLOAT, RHIResourceState::UnorderedAccess);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		auto *rhi_context = renderer->GetRHIContext();

		struct PipelineInfo
		{
			std::shared_ptr<RHIPipelineState> pipeline_state = nullptr;
			ShaderMeta                        shader_meta;
		};

		PipelineInfo ssao_pipeline;
		PipelineInfo blur_pipeline;

		{
			auto shader = renderer->RequireShader("Source/Shaders/AmbientOcclusion/SSAO.hlsl", "SSAO", RHIShaderStage::Compute);

			ssao_pipeline.shader_meta    = renderer->RequireShaderMeta(shader);
			ssao_pipeline.pipeline_state = std::shared_ptr<RHIPipelineState>(std::move(rhi_context->CreatePipelineState()));
			ssao_pipeline.pipeline_state->SetShader(RHIShaderStage::Compute, shader);
		}

		{
			auto shader = renderer->RequireShader("Source/Shaders/AmbientOcclusion/SSAO.hlsl", "SSAOBlur", RHIShaderStage::Compute);

			blur_pipeline.shader_meta    = renderer->RequireShaderMeta(shader);
			blur_pipeline.pipeline_state = std::shared_ptr<RHIPipelineState>(std::move(rhi_context->CreatePipelineState()));
			blur_pipeline.pipeline_state->SetShader(RHIShaderStage::Compute, shader);
		}

		std::default_random_engine            rnd_engine(0);
		std::uniform_real_distribution<float> rnd_dist(0.0f, 1.0f);

		std::shared_ptr<RHIBuffer> sample_buffer = rhi_context->CreateBuffer<glm::vec4>(SSAO_KERNEL_SIZE, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);
		// Update sample buffer
		{
			std::vector<glm::vec4> ssao_kernal(SSAO_KERNEL_SIZE);
			for (uint32_t i = 0; i < SSAO_KERNEL_SIZE; ++i)
			{
				glm::vec3 sample(rnd_dist(rnd_engine) * 2.0 - 1.0, rnd_dist(rnd_engine) * 2.0 - 1.0, rnd_dist(rnd_engine));
				sample = glm::normalize(sample);
				sample *= rnd_dist(rnd_engine);
				float scale    = float(i) / float(SSAO_KERNEL_SIZE);
				scale          = glm::mix(0.1f, 1.0f, scale * scale);
				ssao_kernal[i] = glm::vec4(sample * scale, 0.0f);
			}
			sample_buffer->CopyToDevice(ssao_kernal.data(), ssao_kernal.size() * sizeof(glm::vec4));
		}

		std::shared_ptr<RHITexture> noise_texture = rhi_context->CreateTexture2D(SSAO_NOISE_DIM, SSAO_NOISE_DIM, RHIFormat::R32G32B32A32_FLOAT, RHITextureUsage::Transfer | RHITextureUsage::ShaderResource, false);
		// Update noise texture
		{
			std::vector<glm::vec4> ssao_noise(SSAO_NOISE_DIM * SSAO_NOISE_DIM);
			for (unsigned int i = 0; i < SSAO_NOISE_DIM * SSAO_NOISE_DIM; i++)
			{
				ssao_noise[i] = glm::vec4(rnd_dist(rnd_engine) * 2.0f - 1.0f, rnd_dist(rnd_engine) * 2.0f - 1.0f, 0.0f, 0.0f);
			}
			auto staging_buffer = rhi_context->CreateBuffer<glm::vec4>(SSAO_NOISE_DIM * SSAO_NOISE_DIM, RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU);
			staging_buffer->CopyToDevice(ssao_noise.data(), ssao_noise.size() * sizeof(glm::vec4));
			{
				auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Transfer);
				cmd_buffer->Begin();
				cmd_buffer->ResourceStateTransition({
				                                        TextureStateTransition{noise_texture.get(), RHIResourceState::Undefined, RHIResourceState::TransferDest},
				                                    },
				                                    {});
				cmd_buffer->CopyBufferToTexture(staging_buffer.get(), noise_texture.get(), 0, 0, 1);
				cmd_buffer->ResourceStateTransition({
				                                        TextureStateTransition{noise_texture.get(), RHIResourceState::TransferDest, RHIResourceState::ShaderResource},
				                                    },
				                                    {});
				cmd_buffer->End();
				rhi_context->Execute(cmd_buffer);
			}
		}

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto normal         = render_graph.GetTexture(desc.GetPin("Normal").handle);
			auto position_depth = render_graph.GetTexture(desc.GetPin("PositionDepth").handle);
			auto output         = render_graph.GetTexture(desc.GetPin("Output").handle);

			auto *view = black_board.Get<View>();

			// SSAO
			{
				cmd_buffer->BeginMarker("SSAO");
				auto descriptor = rhi_context->CreateDescriptor(ssao_pipeline.shader_meta);
				descriptor->BindTexture("Normal", normal, RHITextureDimension::Texture2D)
				    .BindTexture("PositionDepth", position_depth, RHITextureDimension::Texture2D)
				    .BindTexture("NoiseTexture", noise_texture.get(), RHITextureDimension::Texture2D)
				    .BindTexture("SSAOMap", output, RHITextureDimension::Texture2D)
				    .BindBuffer("SampleBuffer", sample_buffer.get())
				    .BindBuffer("ViewBuffer", view->buffer.get())
				    .BindSampler("TexSampler", rhi_context->CreateSampler(SamplerDesc::LinearClamp()));
				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(ssao_pipeline.pipeline_state.get());
				cmd_buffer->Dispatch(output->GetDesc().width, output->GetDesc().height, 1, 8, 8, 1);
				cmd_buffer->EndMarker();
			}

			// Sync
			{
				cmd_buffer->ResourceStateTransition({
				                                        TextureStateTransition{output, RHIResourceState::UnorderedAccess, RHIResourceState::UnorderedAccess},
				                                    },
				                                    {});
			}

			// SSAO Blur
			{
				cmd_buffer->BeginMarker("SSAO Blur");
				auto descriptor = rhi_context->CreateDescriptor(blur_pipeline.shader_meta);
				descriptor->BindTexture("SSAOMap", output, RHITextureDimension::Texture2D);
				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(blur_pipeline.pipeline_state.get());
				cmd_buffer->Dispatch(output->GetDesc().width, output->GetDesc().height, 1, 8, 8, 1);
				cmd_buffer->EndMarker();
			}
		};
	}

	virtual void OnImGui(Variant *config)
	{
	}
};

CONFIGURATION_PASS(SSAO)