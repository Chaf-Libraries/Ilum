#include "IPass.hpp"

using namespace Ilum;

#define IRRADIANCE_CUBEMAP_SIZE 128
#define IRRADIANCE_WORK_GROUP_SIZE 8
#define SH_INTERMEDIATE_SIZE (IRRADIANCE_CUBEMAP_SIZE / IRRADIANCE_WORK_GROUP_SIZE)
#define CUBEMAP_FACE_NUM 6
#define PREFILTER_MAP_SIZE 256
#define PREFILTER_MIP_LEVELS 5

class IBL : public RenderPass<IBL>
{
	enum class EnvironmentMapType
	{
		Skybox,
		Procedure
	};

	struct Config
	{
		EnvironmentMapType type = EnvironmentMapType::Skybox;

		size_t hash = 0;
	};

  public:
	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::Rasterization)
		    .SetName("IBL")
		    .SetCategory("Shading")
		    .SetConfig(Config())
		    .WriteTexture2D(handle++, "IrradianceSH", RHIFormat::R32G32B32A32_FLOAT, RHIResourceState::UnorderedAccess, 9, 1)
		    .WriteTexture2D(handle++, "PrefilterMap", RHIFormat::R32G32B32A32_FLOAT, RHIResourceState::UnorderedAccess, PREFILTER_MAP_SIZE, PREFILTER_MAP_SIZE, CUBEMAP_FACE_NUM, PREFILTER_MIP_LEVELS);
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
		PipelineInfo cubemap_prefilter;

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

		{
			auto shader = renderer->RequireShader("Source/Shaders/Shading/IBL.hlsl", "CubmapPrefilter", RHIShaderStage::Compute);

			cubemap_prefilter.pipeline_state = std::shared_ptr<RHIPipelineState>(std::move(rhi_context->CreatePipelineState()));
			cubemap_prefilter.pipeline_state->SetShader(RHIShaderStage::Compute, shader);
			cubemap_prefilter.shader_meta = renderer->RequireShaderMeta(shader);
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
			auto prefilter_map = render_graph.GetTexture(desc.GetPin("PrefilterMap").handle);

			auto   *gpu_scene   = black_board.Get<GPUScene>();
			auto   *view        = black_board.Get<View>();
			Config *config_data = config.Convert<Config>();

			if (gpu_scene->texture.texture_cube)
			{
				if (config_data->hash != Hash(gpu_scene->texture.texture_cube, irradiance_sh, prefilter_map))
				{
					config_data->hash = Hash(gpu_scene->texture.texture_cube, irradiance_sh, prefilter_map);
				}
				else
				{
					return;
				}

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
						descriptor->BindTexture("Skybox", gpu_scene->texture.texture_cube, RHITextureDimension::TextureCube)
						    .BindSampler("SkyboxSampler", rhi_context->CreateSampler(SamplerDesc::LinearClamp()))
						    .BindTexture("SHIntermediate", sh_intermediate.get(), RHITextureDimension::Texture2DArray);
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

				// Specular Prefilter
				{
					cmd_buffer->BeginMarker("Cubemap Prefilter");
					for (uint32_t i = 0; i < PREFILTER_MIP_LEVELS; i++)
					{
						auto *descriptor = rhi_context->CreateDescriptor(cubemap_prefilter.shader_meta);
						descriptor->BindTexture("Skybox", gpu_scene->texture.texture_cube, RHITextureDimension::TextureCube)
						    .BindSampler("SkyboxSampler", rhi_context->CreateSampler(SamplerDesc::LinearClamp()))
						    .BindTexture("PrefilterMap", prefilter_map, TextureRange{RHITextureDimension::Texture2DArray, i, 1, 0, 6});
						cmd_buffer->BindDescriptor(descriptor);
						cmd_buffer->BindPipelineState(cubemap_prefilter.pipeline_state.get());
						cmd_buffer->Dispatch(PREFILTER_MAP_SIZE << i, PREFILTER_MAP_SIZE << i, 6, 8, 8, 1);
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