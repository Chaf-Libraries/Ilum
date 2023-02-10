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
		    .WriteTexture2D(handle++, "Direct Illumination", 0, 0, RHIFormat::R16G16B16A16_FLOAT, RHIResourceState::UnorderedAccess);
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
			auto   *visibility_buffer = render_graph.GetTexture(desc.GetPin("Visibility Buffer").handle);
			auto   *depth_buffer      = render_graph.GetTexture(desc.GetPin("Depth Buffer").handle);
			auto   *output            = render_graph.GetTexture(desc.GetPin("Direct Illumination").handle);
			Config *config_data       = config.Convert<Config>();

			GPUScene   *gpu_scene   = black_board.Get<GPUScene>();
			View       *view        = black_board.Get<View>();
			RHIContext *rhi_context = renderer->GetRHIContext();

			LightingPassData *pass_data   = black_board.Has<LightingPassData>() ? black_board.Get<LightingPassData>() : black_board.Add<LightingPassData>();
			ShadowMapData    *shadow_data = black_board.Has<ShadowMapData>() ? black_board.Get<ShadowMapData>() : nullptr;

			size_t material_count = renderer->GetResourceManager()->GetValidResourceCount<ResourceType::Material>() + 1;

			bool has_mesh              = gpu_scene->mesh_buffer.instance_count != 0;
			bool has_skinned_mesh      = gpu_scene->skinned_mesh_buffer.instance_count != 0;
			bool has_point_light       = gpu_scene->light.info.point_light_count != 0;
			bool has_spot_light        = gpu_scene->light.info.spot_light_count != 0;
			bool has_directional_light = gpu_scene->light.info.directional_light_count != 0;
			bool has_rect_light        = gpu_scene->light.info.rect_light_count != 0;
			bool has_shadow            = gpu_scene->light.has_shadow;

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
			        output,
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

			cmd_buffer->FillTexture(output, RHIResourceState::TransferDest, TextureRange{}, glm::vec4(0.f));
			cmd_buffer->FillBuffer(pass_data->material_offset_buffer.get(), RHIResourceState::TransferDest, pass_data->material_offset_buffer->GetDesc().size);
			cmd_buffer->FillBuffer(pass_data->material_count_buffer.get(), RHIResourceState::TransferDest, pass_data->material_count_buffer->GetDesc().size);
			cmd_buffer->FillBuffer(pass_data->indirect_command_buffer.get(), RHIResourceState::TransferDest, pass_data->indirect_command_buffer->GetDesc().size);

			cmd_buffer->ResourceStateTransition(
			    {TextureStateTransition{
			        output,
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

			// std::vector<uint32_t> debug_data(material_count);
			// pass_data->material_count_buffer->CopyToHost(debug_data.data(), debug_data.size() * sizeof(uint32_t));
			// std::cout << "Material Count Buffer: ";
			// for (auto data : debug_data)
			//{
			//	std::cout << data << " ";
			// }
			// std::cout << std::endl;

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
				    .BindBuffer("MeshInstanceBuffer", gpu_scene->mesh_buffer.instances.get())
				    .BindBuffer("SkinnedMeshInstanceBuffer", gpu_scene->skinned_mesh_buffer.instances.get())
				    .BindBuffer("MaterialCountBuffer", pass_data->material_count_buffer.get());
				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(pipeline_state.get());
				cmd_buffer->Dispatch(visibility_buffer->GetDesc().width, visibility_buffer->GetDesc().height, 1, 8, 8, 1);
				cmd_buffer->EndMarker();
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
				         RHIResourceState::ShaderResource},
				     BufferStateTransition{
				         pass_data->material_count_buffer.get(),
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
				    .BindBuffer("MeshInstanceBuffer", gpu_scene->mesh_buffer.instances.get())
				    .BindBuffer("SkinnedMeshInstanceBuffer", gpu_scene->skinned_mesh_buffer.instances.get())
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
					        has_shadow ? "HAS_SHADOW" : "NO_SHADOW",
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
					    .BindTexture("Textures", gpu_scene->textures.texture_2d, RHITextureDimension::Texture2D)
					    .BindSampler("Samplers", gpu_scene->samplers)
					    .BindBuffer("MaterialOffsets", gpu_scene->material.material_offset.get())
					    .BindBuffer("MaterialBuffer", gpu_scene->material.material_buffer.get())
					    .BindTexture("Output", output, RHITextureDimension::Texture2D);

					if (has_mesh)
					{
						descriptor->BindBuffer("MeshVertexBuffer", gpu_scene->mesh_buffer.vertex_buffers)
						    .BindBuffer("MeshIndexBuffer", gpu_scene->mesh_buffer.index_buffers)
						    .BindBuffer("MeshInstanceBuffer", gpu_scene->mesh_buffer.instances.get());
					}

					if (has_skinned_mesh)
					{
						descriptor->BindBuffer("SkinnedMeshVertexBuffer", gpu_scene->skinned_mesh_buffer.vertex_buffers)
						    .BindBuffer("SkinnedMeshIndexBuffer", gpu_scene->skinned_mesh_buffer.index_buffers)
						    .BindBuffer("SkinnedMeshInstanceBuffer", gpu_scene->skinned_mesh_buffer.instances.get());
					}

					if (has_point_light)
					{
						descriptor->BindBuffer("PointLightBuffer", gpu_scene->light.point_light_buffer.get());
						if (has_shadow)
						{
							descriptor->BindTexture("PointLightShadow", shadow_data->omni_shadow_map.get(), RHITextureDimension::TextureCubeArray);
						}
					}

					if (has_spot_light)
					{
						descriptor->BindBuffer("SpotLightBuffer", gpu_scene->light.spot_light_buffer.get());
						if (has_shadow)
						{
							descriptor->BindTexture("SpotLightShadow", shadow_data->shadow_map.get(), RHITextureDimension::Texture2DArray);
						}
					}

					if (has_directional_light)
					{
						descriptor->BindBuffer("DirectionalLightBuffer", gpu_scene->light.directional_light_buffer.get());
						if (has_shadow)
						{
							descriptor->BindTexture("DirectionalLightShadow", shadow_data->cascade_shadow_map.get(), RHITextureDimension::Texture2DArray);
						}
					}

					if (has_rect_light)
					{
						descriptor->BindBuffer("RectLightBuffer", gpu_scene->light.rect_light_buffer.get());
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
