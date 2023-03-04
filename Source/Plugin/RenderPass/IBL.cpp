#include "IPass.hpp"

using namespace Ilum;

#define IRRADIANCE_CUBEMAP_SIZE 128
#define IRRADIANCE_WORK_GROUP_SIZE 8
#define SH_INTERMEDIATE_SIZE (IRRADIANCE_CUBEMAP_SIZE / IRRADIANCE_WORK_GROUP_SIZE)
#define CUBEMAP_FACE_NUM 6

class IBL : public RenderPass<IBL>
{
	enum class EnvironmentMapType
	{
		Skybox,
		Procedure
	};

	struct Config
	{
		glm::uvec2 extent;

		EnvironmentMapType type = EnvironmentMapType::Skybox;
	};

  public:
	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::Rasterization)
		    .SetName("IBL")
		    .SetCategory("Shading")
		    .SetConfig(Config())
		    .WriteTexture2D(handle++, "IrradianceSH", 9, 1, RHIFormat::R32G32B32A32_FLOAT, RHIResourceState::UnorderedAccess);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		auto *rhi_context = renderer->GetRHIContext();

		struct PipelineInfo
		{
			std::shared_ptr<RHIPipelineState> pipeline_state = nullptr;

			ShaderMeta shader_meta;
		};

		PipelineInfo cubemap_sh_projection;
		PipelineInfo cubemap_sh_add;

		{
			auto shader = renderer->RequireShader("Source/Shaders/Shading/IBL.hlsl", "CubemapSHProjection", RHIShaderStage::Compute);

			cubemap_sh_projection.pipeline_state = std::shared_ptr<RHIPipelineState>(std::move(rhi_context->CreatePipelineState()));
			cubemap_sh_projection.pipeline_state->SetShader(RHIShaderStage::Compute, shader);
			cubemap_sh_projection.shader_meta = renderer->RequireShaderMeta(shader);
		}

		{
			auto shader = renderer->RequireShader("Source/Shaders/Shading/IBL.hlsl", "CubemapSHAdd", RHIShaderStage::Compute);

			cubemap_sh_add.pipeline_state = std::shared_ptr<RHIPipelineState>(std::move(rhi_context->CreatePipelineState()));
			cubemap_sh_add.pipeline_state->SetShader(RHIShaderStage::Compute, shader);
			cubemap_sh_add.shader_meta = renderer->RequireShaderMeta(shader);
		}

		std::shared_ptr<RHIBuffer>  config_buffer   = std::move(rhi_context->CreateBuffer<Config>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU));
		std::shared_ptr<RHITexture> sh_intermediate = std::move(rhi_context->CreateTexture2DArray(SH_INTERMEDIATE_SIZE * 9, SH_INTERMEDIATE_SIZE, 6, RHIFormat::R32G32B32A32_FLOAT, RHITextureUsage::UnorderedAccess | RHITextureUsage::ShaderResource, false));

		{
			auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Compute);
			cmd_buffer->Begin();
			cmd_buffer->ResourceStateTransition({
			                                        TextureStateTransition{
			                                            sh_intermediate.get(),
			                                            RHIResourceState::Undefined,
			                                            RHIResourceState::ShaderResource,
			                                        },
			                                    },
			                                    {});
			cmd_buffer->End();
			rhi_context->Execute(cmd_buffer);
		}

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto irradiance_sh = render_graph.GetTexture(desc.GetPin("IrradianceSH").handle);

			auto   *gpu_scene   = black_board.Get<GPUScene>();
			auto   *view        = black_board.Get<View>();
			Config *config_data = config.Convert<Config>();

			config_data->extent = {1024, 1024};

			if (gpu_scene->textures.texture_cube)
			{
				config_buffer->CopyToDevice(config_data, sizeof(Config));

				// Diffuse Environment PRT
				{
					cmd_buffer->BeginMarker("Diffuse Environment PRT");

					// Resource Transition
					{
						cmd_buffer->ResourceStateTransition({
						                                        TextureStateTransition{
						                                            sh_intermediate.get(),
						                                            RHIResourceState::ShaderResource,
						                                            RHIResourceState::UnorderedAccess,
						                                        },
						                                    },
						                                    {});
					}

					// Cubemap SH Projection
					{
						cmd_buffer->BeginMarker("Cubemap SH Projection");
						auto descriptor = rhi_context->CreateDescriptor(cubemap_sh_projection.shader_meta);
						descriptor->BindTexture("Skybox", gpu_scene->textures.texture_cube, RHITextureDimension::TextureCube)
						    .BindSampler("SkyboxSampler", rhi_context->CreateSampler(SamplerDesc::LinearClamp()))
						    .BindTexture("SHIntermediate", sh_intermediate.get(), RHITextureDimension::Texture2DArray)
						    .BindBuffer("ConfigBuffer", config_buffer.get());
						cmd_buffer->BindDescriptor(descriptor);
						cmd_buffer->BindPipelineState(cubemap_sh_projection.pipeline_state.get());
						cmd_buffer->Dispatch(IRRADIANCE_CUBEMAP_SIZE, IRRADIANCE_CUBEMAP_SIZE, 6, IRRADIANCE_WORK_GROUP_SIZE, IRRADIANCE_WORK_GROUP_SIZE, 1);
						cmd_buffer->EndMarker();
					}

					// Resource Transition
					{
						cmd_buffer->ResourceStateTransition({
						                                        TextureStateTransition{
						                                            sh_intermediate.get(),
						                                            RHIResourceState::UnorderedAccess,
						                                            RHIResourceState::ShaderResource,
						                                        },
						                                    },
						                                    {});
					}

					// Cubemap SH Add
					{
						cmd_buffer->BeginMarker("Cubemap SH Add");
						auto descriptor = rhi_context->CreateDescriptor(cubemap_sh_add.shader_meta);
						descriptor->BindTexture("IrradianceSH", irradiance_sh, RHITextureDimension::Texture2D)
						    .BindTexture("SHIntermediate", sh_intermediate.get(), RHITextureDimension::Texture2DArray);
						cmd_buffer->BindDescriptor(descriptor);
						cmd_buffer->BindPipelineState(cubemap_sh_add.pipeline_state.get());
						cmd_buffer->Dispatch(9, SH_INTERMEDIATE_SIZE, CUBEMAP_FACE_NUM, 1, SH_INTERMEDIATE_SIZE, CUBEMAP_FACE_NUM);
						cmd_buffer->EndMarker();
					}
					cmd_buffer->EndMarker();
				}
			}
		};
	}

	virtual void OnImGui(Variant *config)
	{
	}
};

CONFIGURATION_PASS(IBL)