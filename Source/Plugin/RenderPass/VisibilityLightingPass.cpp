#include "IPass.hpp"
#include "PassData.hpp"

#include <Material/MaterialData.hpp>
#include <Resource/ResourceManager.hpp>

using namespace Ilum;

class VisibilityLightingPass : public RenderPass<VisibilityLightingPass>
{
	enum class ShadowFilterMode
	{
		None,
		Hard,
		PCF,
		PCSS
	};

	struct Config
	{
		ShadowFilterMode shadow_filter_mode = ShadowFilterMode::PCF;
	};

	std::unordered_map<ShadowFilterMode, const char *> shadow_filter_modes = {
	    {ShadowFilterMode::None, "SHADOW_FILTER_NONE"},
	    {ShadowFilterMode::Hard, "SHADOW_FILTER_HARD"},
	    {ShadowFilterMode::PCF, "SHADOW_FILTER_PCF"},
	    {ShadowFilterMode::PCSS, "SHADOW_FILTER_PCSS"},
	};

  public:
	VisibilityLightingPass() = default;

	~VisibilityLightingPass() = default;

	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::Compute)
		    .SetName("VisibilityLightingPass")
		    .SetCategory("RenderPath")
		    .SetConfig(Config())
		    .ReadTexture2D(handle++, "Visibility Buffer", RHIResourceState::ShaderResource)
		    .ReadTexture2D(handle++, "Depth Buffer", RHIResourceState::ShaderResource)
		    .ReadTexture2D(handle++, "ShadowMap", RHIResourceState::ShaderResource)
		    .ReadTexture2D(handle++, "CascadeShadowMap", RHIResourceState::ShaderResource)
		    .ReadTexture2D(handle++, "OmniShadowMap", RHIResourceState::ShaderResource)
		    .ReadTexture2D(handle++, "IrradianceSH", RHIResourceState::ShaderResource)
		    .ReadTexture2D(handle++, "PrefilterMap", RHIResourceState::ShaderResource)
		    .WriteTexture2D(handle++, "Position Depth", RHIFormat::R32G32B32A32_FLOAT, RHIResourceState::UnorderedAccess)
		    .WriteTexture2D(handle++, "Normal Roughness", RHIFormat::R8G8B8A8_UNORM, RHIResourceState::UnorderedAccess)
		    .WriteTexture2D(handle++, "Albedo Metallic", RHIFormat::R8G8B8A8_UNORM, RHIResourceState::UnorderedAccess)
		    .WriteTexture2D(handle++, "Env DI", RHIFormat::R16G16B16A16_FLOAT, RHIResourceState::UnorderedAccess)
		    .WriteTexture2D(handle++, "Light DI", RHIFormat::R16G16B16A16_FLOAT, RHIResourceState::UnorderedAccess);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		std::shared_ptr<RHIPipelineState> pipeline_state = std::move(renderer->GetRHIContext()->CreatePipelineState());

		struct LightingPassData
		{
			std::unique_ptr<RHIBuffer> material_count_buffer   = nullptr;
			std::unique_ptr<RHIBuffer> material_offset_buffer  = nullptr;
			std::unique_ptr<RHIBuffer> material_pixel_buffer   = nullptr;
			std::unique_ptr<RHIBuffer> indirect_command_buffer = nullptr;
		};

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto *visibility_buffer         = render_graph.GetTexture(desc.GetPin("Visibility Buffer").handle);
			auto *depth_buffer              = render_graph.GetTexture(desc.GetPin("Depth Buffer").handle);
			auto *shadow_map                = render_graph.GetTexture(desc.GetPin("ShadowMap").handle);
			auto *cascade_shadow_map        = render_graph.GetTexture(desc.GetPin("CascadeShadowMap").handle);
			auto *omni_shadow_map           = render_graph.GetTexture(desc.GetPin("OmniShadowMap").handle);
			auto *irradiance_sh             = render_graph.GetTexture(desc.GetPin("IrradianceSH").handle);
			auto *prefilter_map             = render_graph.GetTexture(desc.GetPin("PrefilterMap").handle);
			auto *env_direct_illumination   = render_graph.GetTexture(desc.GetPin("Env DI").handle);
			auto *light_direct_illumination = render_graph.GetTexture(desc.GetPin("Light DI").handle);
			auto *position_depth            = render_graph.GetTexture(desc.GetPin("Position Depth").handle);
			auto *normal_roughness          = render_graph.GetTexture(desc.GetPin("Normal Roughness").handle);
			auto *albedo_metallic           = render_graph.GetTexture(desc.GetPin("Albedo Metallic").handle);

			Config *config_data = config.Convert<Config>();

			GPUScene   *gpu_scene   = black_board.Get<GPUScene>();
			View       *view        = black_board.Get<View>();
			RHIContext *rhi_context = renderer->GetRHIContext();

			LightingPassData *pass_data = black_board.Has<LightingPassData>() ? black_board.Get<LightingPassData>() : black_board.Add<LightingPassData>();

			size_t material_count = renderer->GetResourceManager()->GetValidResourceCount<ResourceType::Material>() + 1;

			bool has_mesh              = gpu_scene->opaque_mesh.instance_count != 0;
			bool has_skinned_mesh      = gpu_scene->opaque_skinned_mesh.instance_count != 0;
			bool has_point_light       = gpu_scene->light.info.point_light_count != 0;
			bool has_spot_light        = gpu_scene->light.info.spot_light_count != 0;
			bool has_directional_light = gpu_scene->light.info.directional_light_count != 0;
			bool has_rect_light        = gpu_scene->light.info.rect_light_count != 0;
			bool has_env_light         = gpu_scene->texture.texture_cube != nullptr;

			if (!pass_data->material_count_buffer ||
			    pass_data->material_count_buffer->GetDesc().size != material_count * sizeof(uint32_t))
			{
				pass_data->material_count_buffer = rhi_context->CreateBuffer<uint32_t>(material_count, RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
			}

			if (!pass_data->material_offset_buffer ||
			    pass_data->material_offset_buffer->GetDesc().size != material_count * sizeof(uint32_t))
			{
				pass_data->material_offset_buffer = rhi_context->CreateBuffer<uint32_t>(material_count, RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
			}

			if (!pass_data->material_pixel_buffer ||
			    pass_data->material_pixel_buffer->GetDesc().size != static_cast<size_t>(visibility_buffer->GetDesc().width * visibility_buffer->GetDesc().height) * sizeof(uint32_t))
			{
				pass_data->material_pixel_buffer = rhi_context->CreateBuffer<uint32_t>(visibility_buffer->GetDesc().width * visibility_buffer->GetDesc().height, RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
			}

			if (!pass_data->indirect_command_buffer ||
			    pass_data->indirect_command_buffer->GetDesc().size != material_count * sizeof(RHIDispatchIndirectCommand))
			{
				pass_data->indirect_command_buffer = rhi_context->CreateBuffer<RHIDispatchIndirectCommand>(material_count, RHIBufferUsage::Indirect | RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
				cmd_buffer->ResourceStateTransition(
				    {},
				    {BufferStateTransition{
				        pass_data->indirect_command_buffer.get(),
				        RHIResourceState::Undefined,
				        RHIResourceState::IndirectBuffer}});
			}

			cmd_buffer->ResourceStateTransition(
			    {TextureStateTransition{
			         light_direct_illumination,
			         RHIResourceState::UnorderedAccess,
			         RHIResourceState::TransferDest},
			     TextureStateTransition{
			         position_depth,
			         RHIResourceState::UnorderedAccess,
			         RHIResourceState::TransferDest},
			     TextureStateTransition{
			         normal_roughness,
			         RHIResourceState::UnorderedAccess,
			         RHIResourceState::TransferDest},
			     TextureStateTransition{
			         albedo_metallic,
			         RHIResourceState::UnorderedAccess,
			         RHIResourceState::TransferDest}},
			    {BufferStateTransition{
			         pass_data->indirect_command_buffer.get(),
			         RHIResourceState::IndirectBuffer,
			         RHIResourceState::TransferDest},
			     BufferStateTransition{
			         pass_data->material_offset_buffer.get(),
			         RHIResourceState::UnorderedAccess,
			         RHIResourceState::TransferDest},
			     BufferStateTransition{
			         pass_data->material_count_buffer.get(),
			         RHIResourceState::UnorderedAccess,
			         RHIResourceState::TransferDest}});

			cmd_buffer->FillTexture(light_direct_illumination, RHIResourceState::TransferDest, TextureRange{}, glm::vec4(0.f));
			cmd_buffer->FillTexture(position_depth, RHIResourceState::TransferDest, TextureRange{}, glm::vec4(0.f));
			cmd_buffer->FillTexture(normal_roughness, RHIResourceState::TransferDest, TextureRange{}, glm::vec4(0.f));
			cmd_buffer->FillTexture(albedo_metallic, RHIResourceState::TransferDest, TextureRange{}, glm::vec4(0.f));
			cmd_buffer->FillBuffer(pass_data->material_offset_buffer.get(), RHIResourceState::TransferDest, pass_data->material_offset_buffer->GetDesc().size);
			cmd_buffer->FillBuffer(pass_data->material_count_buffer.get(), RHIResourceState::TransferDest, pass_data->material_count_buffer->GetDesc().size);
			cmd_buffer->FillBuffer(pass_data->indirect_command_buffer.get(), RHIResourceState::TransferDest, pass_data->indirect_command_buffer->GetDesc().size);

			cmd_buffer->ResourceStateTransition(
			    {TextureStateTransition{
			         light_direct_illumination,
			         RHIResourceState::TransferDest,
			         RHIResourceState::UnorderedAccess},
			     TextureStateTransition{
			         position_depth,
			         RHIResourceState::TransferDest,
			         RHIResourceState::UnorderedAccess},
			     TextureStateTransition{
			         normal_roughness,
			         RHIResourceState::TransferDest,
			         RHIResourceState::UnorderedAccess},
			     TextureStateTransition{
			         albedo_metallic,
			         RHIResourceState::TransferDest,
			         RHIResourceState::UnorderedAccess}},
			    {BufferStateTransition{
			         pass_data->material_offset_buffer.get(),
			         RHIResourceState::TransferDest,
			         RHIResourceState::UnorderedAccess},
			     BufferStateTransition{
			         pass_data->indirect_command_buffer.get(),
			         RHIResourceState::TransferDest,
			         RHIResourceState::UnorderedAccess},
			     BufferStateTransition{
			         pass_data->material_count_buffer.get(),
			         RHIResourceState::TransferDest,
			         RHIResourceState::UnorderedAccess}});

			// Collect material count
			{
				cmd_buffer->BeginMarker("Collect Material Count");
				auto *shader = renderer->RequireShader(
				    "Source/Shaders/RenderPath/VisibilityLightingPass.hlsl",
				    "CollectMaterialCount",
				    RHIShaderStage::Compute,
				    {
				        has_mesh ? "HAS_MESH" : "NO_MESH",
				        has_skinned_mesh ? "HAS_SKINNED_MESH" : "NO_SKINNED_MESH",
				    });
				auto meta = renderer->RequireShaderMeta(shader);
				pipeline_state->ClearShader().SetShader(RHIShaderStage::Compute, shader);
				auto descriptor = rhi_context->CreateDescriptor(meta);
				descriptor->BindTexture("VisibilityBuffer", visibility_buffer, RHITextureDimension::Texture2D)
				    .BindTexture("DepthBuffer", depth_buffer, RHITextureDimension::Texture2D)
				    .BindBuffer("MeshInstanceBuffer", gpu_scene->opaque_mesh.instances.get())
				    .BindBuffer("SkinnedMeshInstanceBuffer", gpu_scene->opaque_skinned_mesh.instances.get())
				    .BindBuffer("MaterialCountBuffer", pass_data->material_count_buffer.get());
				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(pipeline_state.get());
				cmd_buffer->Dispatch(visibility_buffer->GetDesc().width, visibility_buffer->GetDesc().height, 1, 8, 8, 1);
				cmd_buffer->EndMarker();
			}

			{
				cmd_buffer->ResourceStateTransition(
				    {},
				    {BufferStateTransition{
				        pass_data->material_count_buffer.get(),
				        RHIResourceState::UnorderedAccess,
				        RHIResourceState::ShaderResource}});
			}

			// Calculate material offset
			{
				cmd_buffer->BeginMarker("Calculate Material Offset");
				auto *shader = renderer->RequireShader(
				    "Source/Shaders/RenderPath/VisibilityLightingPass.hlsl",
				    "CalculateMaterialOffset",
				    RHIShaderStage::Compute);
				auto meta = renderer->RequireShaderMeta(shader);
				pipeline_state->ClearShader().SetShader(RHIShaderStage::Compute, shader);
				auto descriptor = rhi_context->CreateDescriptor(meta);
				descriptor->BindBuffer("MaterialCountBuffer", pass_data->material_count_buffer.get())
				    .BindBuffer("MaterialOffsetBuffer", pass_data->material_offset_buffer.get());
				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(pipeline_state.get());
				cmd_buffer->Dispatch(static_cast<uint32_t>(material_count), 1, 1, 128, 1, 1);
				cmd_buffer->EndMarker();
			}

			// Write -> Read
			{
				cmd_buffer->ResourceStateTransition(
				    {},
				    {BufferStateTransition{
				        pass_data->material_offset_buffer.get(),
				        RHIResourceState::UnorderedAccess,
				        RHIResourceState::ShaderResource}});
			}

			// Calculate pixel buffer
			{
				cmd_buffer->BeginMarker("Calculate Pixel Buffer");
				auto *shader = renderer->RequireShader(
				    "Source/Shaders/RenderPath/VisibilityLightingPass.hlsl",
				    "CalculatePixelBuffer",
				    RHIShaderStage::Compute,
				    {
				        has_mesh ? "HAS_MESH" : "NO_MESH",
				        has_skinned_mesh ? "HAS_SKINNED_MESH" : "NO_SKINNED_MESH",
				    });
				auto meta = renderer->RequireShaderMeta(shader);
				pipeline_state->ClearShader().SetShader(RHIShaderStage::Compute, shader);
				auto descriptor = rhi_context->CreateDescriptor(meta);
				descriptor->BindTexture("VisibilityBuffer", visibility_buffer, RHITextureDimension::Texture2D)
				    .BindTexture("DepthBuffer", depth_buffer, RHITextureDimension::Texture2D)
				    .BindBuffer("MeshInstanceBuffer", gpu_scene->opaque_mesh.instances.get())
				    .BindBuffer("SkinnedMeshInstanceBuffer", gpu_scene->opaque_skinned_mesh.instances.get())
				    .BindBuffer("MaterialCountBuffer", pass_data->material_count_buffer.get())
				    .BindBuffer("MaterialOffsetBuffer", pass_data->material_offset_buffer.get())
				    .BindBuffer("MaterialPixelBuffer", pass_data->material_pixel_buffer.get())
				    .BindBuffer("IndirectCommandBuffer", pass_data->indirect_command_buffer.get());
				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(pipeline_state.get());
				cmd_buffer->Dispatch(visibility_buffer->GetDesc().width, visibility_buffer->GetDesc().height, 1, 8, 8, 1);
				cmd_buffer->EndMarker();
			}

			{
				cmd_buffer->ResourceStateTransition(
				    {},
				    {BufferStateTransition{
				        pass_data->indirect_command_buffer.get(),
				        RHIResourceState::UnorderedAccess,
				        RHIResourceState::UnorderedAccess}});
			}

			// Calculate indirect argument
			{
				cmd_buffer->BeginMarker("Calculate Indirect Argument");
				auto *shader = renderer->RequireShader(
				    "Source/Shaders/RenderPath/VisibilityLightingPass.hlsl",
				    "CalculateIndirectArgument",
				    RHIShaderStage::Compute);
				auto meta = renderer->RequireShaderMeta(shader);
				pipeline_state->ClearShader().SetShader(RHIShaderStage::Compute, shader);
				auto descriptor = rhi_context->CreateDescriptor(meta);
				descriptor->BindBuffer("IndirectCommandBuffer", pass_data->indirect_command_buffer.get());
				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(pipeline_state.get());
				cmd_buffer->Dispatch(static_cast<uint32_t>(material_count), 1, 1, 8, 1, 1);
				cmd_buffer->EndMarker();
			}

			// Write -> Read
			{
				cmd_buffer->ResourceStateTransition(
				    {},
				    {BufferStateTransition{
				         pass_data->material_pixel_buffer.get(),
				         RHIResourceState::UnorderedAccess,
				         RHIResourceState::ShaderResource},
				     BufferStateTransition{
				         pass_data->indirect_command_buffer.get(),
				         RHIResourceState::UnorderedAccess,
				         RHIResourceState::IndirectBuffer}});
			}

			// Dispatch
			{
				cmd_buffer->BeginMarker("Dispatch Indirect");
				for (size_t i = 0; i < material_count; i++)
				{
					auto *shader = renderer->RequireShader(
					    "Source/Shaders/RenderPath/VisibilityLightingPass.hlsl",
					    "DispatchIndirect",
					    RHIShaderStage::Compute,
					    {
					        has_mesh ? "HAS_MESH" : "NO_MESH",
					        has_skinned_mesh ? "HAS_SKINNED_MESH" : "NO_SKINNED_MESH",
					        has_point_light ? "HAS_POINT_LIGHT" : "NO_POINT_LIGHT",
					        has_spot_light ? "HAS_SPOT_LIGHT" : "NO_SPOT_LIGHT",
					        has_directional_light ? "HAS_DIRECTIONAL_LIGHT" : "NO_DIRECTIONAL_LIGHT",
					        has_rect_light ? "HAS_RECT_LIGHT" : "NO_RECT_LIGHT",
					        has_env_light ? "HAS_ENV_LIGHT" : "NO_ENV_LIGHT",
					        shadow_map ? "HAS_SHADOW_MAP" : "NO_SHADOW_MAP",
					        cascade_shadow_map ? "HAS_CASCADE_SHADOW_MAP" : "NO_CASCADE_SHADOW_MAP",
					        omni_shadow_map ? "HAS_OMNI_SHADOW_MAP" : "NO_OMNI_SHADOW_MAP",
					        irradiance_sh ? "HAS_IRRADIANCE_SH" : "NO_IRRADIANCE_SH",
					        prefilter_map ? "HAS_PREFILTER_MAP" : "NO_PREFILTER_MAP",
					        shadow_filter_modes.at(config_data->shadow_filter_mode),
					        "DISPATCH_INDIRECT",
					        "MATERIAL_ID=" + std::to_string(i),
					        i == 0 ? "DEFAULT_MATERIAL" : gpu_scene->material.data[i - 1]->signature,
					    },
					    {
					        i == 0 ? "../Material/Material.hlsli" : gpu_scene->material.data[i - 1]->shader,
					    });
					auto meta = renderer->RequireShaderMeta(shader);
					pipeline_state->ClearShader().SetShader(RHIShaderStage::Compute, shader);
					auto descriptor = rhi_context->CreateDescriptor(meta);
					descriptor->BindTexture("VisibilityBuffer", visibility_buffer, RHITextureDimension::Texture2D)
					    .BindBuffer("ViewBuffer", view->buffer.get())
					    .BindBuffer("LightInfoBuffer", gpu_scene->light.light_info_buffer.get())
					    .BindSampler("ShadowMapSampler", rhi_context->CreateSampler(SamplerDesc::LinearClamp()))
					    .BindBuffer("MaterialPixelBuffer", pass_data->material_pixel_buffer.get())
					    .BindBuffer("MaterialCountBuffer", pass_data->material_count_buffer.get())
					    .BindBuffer("MaterialOffsetBuffer", pass_data->material_offset_buffer.get())
					    .BindTexture("Textures", gpu_scene->texture.texture_2d, RHITextureDimension::Texture2D)
					    .BindSampler("Samplers", gpu_scene->samplers)
					    .BindBuffer("MaterialOffsets", gpu_scene->material.material_offset.get())
					    .BindBuffer("MaterialBuffer", gpu_scene->material.material_buffer.get())
					    .BindTexture("LightDirectIllumination", light_direct_illumination, RHITextureDimension::Texture2D)
					    .BindTexture("EnvDirectIllumination", env_direct_illumination, RHITextureDimension::Texture2D)
					    .BindTexture("PositionDepth", position_depth, RHITextureDimension::Texture2D)
					    .BindTexture("NormalRoughness", normal_roughness, RHITextureDimension::Texture2D)
					    .BindTexture("AlbedoMetallic", albedo_metallic, RHITextureDimension::Texture2D);

					if (has_mesh)
					{
						descriptor->BindBuffer("MeshVertexBuffer", gpu_scene->mesh_buffer.vertex_buffers)
						    .BindBuffer("MeshIndexBuffer", gpu_scene->mesh_buffer.index_buffers)
						    .BindBuffer("MeshInstanceBuffer", gpu_scene->opaque_mesh.instances.get());
					}

					if (has_skinned_mesh)
					{
						descriptor->BindBuffer("SkinnedMeshVertexBuffer", gpu_scene->skinned_mesh_buffer.vertex_buffers)
						    .BindBuffer("SkinnedMeshIndexBuffer", gpu_scene->skinned_mesh_buffer.index_buffers)
						    .BindBuffer("BoneMatrices", gpu_scene->animation.bone_matrics)
						    .BindBuffer("SkinnedMeshInstanceBuffer", gpu_scene->opaque_skinned_mesh.instances.get());
					}

					if (has_point_light)
					{
						descriptor->BindBuffer("PointLightBuffer", gpu_scene->light.point_light_buffer.get());
						if (omni_shadow_map)
						{
							descriptor->BindTexture("PointLightShadow", omni_shadow_map, RHITextureDimension::TextureCubeArray);
						}
					}

					if (has_spot_light)
					{
						descriptor->BindBuffer("SpotLightBuffer", gpu_scene->light.spot_light_buffer.get());
						if (shadow_map)
						{
							descriptor->BindTexture("SpotLightShadow", shadow_map, RHITextureDimension::Texture2DArray);
						}
					}

					if (has_directional_light)
					{
						descriptor->BindBuffer("DirectionalLightBuffer", gpu_scene->light.directional_light_buffer.get());
						if (cascade_shadow_map)
						{
							descriptor->BindTexture("DirectionalLightShadow", cascade_shadow_map, RHITextureDimension::Texture2DArray);
						}
					}

					if (has_rect_light)
					{
						descriptor->BindBuffer("RectLightBuffer", gpu_scene->light.rect_light_buffer.get());
					}

					if (has_env_light)
					{
						if (irradiance_sh)
						{
							descriptor->BindTexture("IrradianceSH", irradiance_sh, RHITextureDimension::Texture2D);
						}
						if (prefilter_map)
						{
							descriptor->BindTexture("PrefilterMap", prefilter_map, RHITextureDimension::TextureCube)
							    .BindSampler("PrefilterMapSampler", rhi_context->CreateSampler(SamplerDesc::LinearClamp()))
							    .BindTexture("GGXPreintegration", black_board.Get<LUT>()->ggx.get(), RHITextureDimension::Texture2D);
						}
					}

					cmd_buffer->BindDescriptor(descriptor);
					cmd_buffer->BindPipelineState(pipeline_state.get());
					cmd_buffer->DispatchIndirect(pass_data->indirect_command_buffer.get(), i * sizeof(RHIDispatchIndirectCommand));
				}
				cmd_buffer->EndMarker();
			}

			cmd_buffer->ResourceStateTransition(
			    {},
			    {BufferStateTransition{
			         pass_data->indirect_command_buffer.get(),
			         RHIResourceState::IndirectBuffer,
			         RHIResourceState::UnorderedAccess},
			     BufferStateTransition{
			         pass_data->material_offset_buffer.get(),
			         RHIResourceState::ShaderResource,
			         RHIResourceState::UnorderedAccess},
			     BufferStateTransition{
			         pass_data->material_pixel_buffer.get(),
			         RHIResourceState::ShaderResource,
			         RHIResourceState::UnorderedAccess},
			     BufferStateTransition{
			         pass_data->material_count_buffer.get(),
			         RHIResourceState::ShaderResource,
			         RHIResourceState::UnorderedAccess}});
		};
	}

	virtual void OnImGui(Variant *config)
	{
		Config *config_data = config->Convert<Config>();

		const char *const shadow_filter_modes[] = {"None", "Hard", "PCF", "PCSS"};
		ImGui::Combo("Shadow Filter Mode", reinterpret_cast<int32_t *>(&config_data->shadow_filter_mode), shadow_filter_modes, 4);
	}
};

CONFIGURATION_PASS(VisibilityLightingPass)
