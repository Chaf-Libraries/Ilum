#include "IPass.hpp"
#include "PassData.hpp"

#include <Resource/Resource/Mesh.hpp>
#include <Resource/Resource/SkinnedMesh.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Components/AllComponents.hpp>
#include <Scene/Scene.hpp>

using namespace Ilum;

class ShadowMapPass : public RenderPass<ShadowMapPass>
{
	struct PipelineDesc
	{
		std::shared_ptr<RHIPipelineState> pipeline = nullptr;

		ShaderMeta meta;
	};

	enum class ShadowMapResolution
	{
		VeryLow,
		Low,
		Medium,
		High,
		VeryHigh
	};

	const std::unordered_map<ShadowMapResolution, uint32_t> shadow_map_resolution =
	    {
	        {ShadowMapResolution::VeryHigh, 1024},
	        {ShadowMapResolution::High, 768},
	        {ShadowMapResolution::Medium, 512},
	        {ShadowMapResolution::Low, 384},
	        {ShadowMapResolution::VeryLow, 256},
	    };

	struct Config
	{
		ShadowMapResolution shadow_map_resolution         = ShadowMapResolution::Medium;
		ShadowMapResolution cascade_shadow_map_resolution = ShadowMapResolution::Medium;
		ShadowMapResolution omni_map_resolution           = ShadowMapResolution::Medium;

		float bias  = 16.f;
		float slope = 4.5f;
	};

  public:
	ShadowMapPass() = default;

	~ShadowMapPass() = default;

	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::Rasterization)
		    .SetConfig(Config())
		    .SetName("ShadowMapPass")
		    .SetCategory("Shading");
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		auto mesh_shadow_pipeline                 = CreatePipeline("Source/Shaders/Shading/Shadow/ShadowMap.hlsl", renderer, false);
		auto skinned_mesh_shadow_pipeline         = CreatePipeline("Source/Shaders/Shading/Shadow/ShadowMap.hlsl", renderer, true);
		auto mesh_cascade_shadow_pipeline         = CreatePipeline("Source/Shaders/Shading/Shadow/CascadeShadowMap.hlsl", renderer, false);
		auto skinned_mesh_cascade_shadow_pipeline = CreatePipeline("Source/Shaders/Shading/Shadow/CascadeShadowMap.hlsl", renderer, true);
		auto mesh_omni_shadow_pipeline            = CreatePipeline("Source/Shaders/Shading/Shadow/OmniShadowMap.hlsl", renderer, false);
		auto skinned_mesh_omni_shadow_pipeline    = CreatePipeline("Source/Shaders/Shading/Shadow/OmniShadowMap.hlsl", renderer, true);

		std::shared_ptr<RHIRenderTarget> shadowmap_render_target         = std::shared_ptr<RHIRenderTarget>(std::move(renderer->GetRHIContext()->CreateRenderTarget()));
		std::shared_ptr<RHIRenderTarget> cascade_shadowmap_render_target = std::shared_ptr<RHIRenderTarget>(std::move(renderer->GetRHIContext()->CreateRenderTarget()));
		std::shared_ptr<RHIRenderTarget> omni_shadowmap_render_target    = std::shared_ptr<RHIRenderTarget>(std::move(renderer->GetRHIContext()->CreateRenderTarget()));

		renderer->GetRenderGraphBlackboard().Get<GPUScene>()->light.has_shadow = true;

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto   *rhi_context = renderer->GetRHIContext();
			auto   *scene       = renderer->GetScene();
			auto   *gpu_scene   = black_board.Get<GPUScene>();
			Config *config_data = config.Convert<Config>();

			auto shadow_map_data = !black_board.Has<ShadowMapData>() ? black_board.Add<ShadowMapData>() : black_board.Get<ShadowMapData>();

			RenderShadowMap(cmd_buffer, mesh_shadow_pipeline, skinned_mesh_shadow_pipeline, renderer, scene, gpu_scene, config_data, shadow_map_data, shadowmap_render_target.get());
			RenderCascadeShadowMap(cmd_buffer, mesh_cascade_shadow_pipeline, skinned_mesh_cascade_shadow_pipeline, renderer, scene, gpu_scene, config_data, shadow_map_data, cascade_shadowmap_render_target.get());
			RenderOmniShadowMap(cmd_buffer, mesh_omni_shadow_pipeline, skinned_mesh_omni_shadow_pipeline, renderer, scene, gpu_scene, config_data, shadow_map_data, omni_shadowmap_render_target.get());
		};
	}

	PipelineDesc CreatePipeline(const std::string &path, Renderer *renderer, bool has_skinned)
	{
		PipelineDesc pipeline_desc;

		pipeline_desc.pipeline = std::shared_ptr<RHIPipelineState>(std::move(renderer->GetRHIContext()->CreatePipelineState()));

		RasterizationState rasterization_state;
		rasterization_state.cull_mode  = RHICullMode::None;
		rasterization_state.front_face = RHIFrontFace::Clockwise;
		pipeline_desc.pipeline->SetRasterizationState(rasterization_state);

		DepthStencilState depth_stencil_state  = {};
		depth_stencil_state.depth_write_enable = true;
		depth_stencil_state.depth_test_enable  = true;
		pipeline_desc.pipeline->SetDepthStencilState(depth_stencil_state);

		if (renderer->GetRHIContext()->IsFeatureSupport(RHIFeature::MeshShading))
		{
			auto *task_shader = renderer->RequireShader(path, "ASmain", RHIShaderStage::Task, {has_skinned ? "HAS_SKINNED" : "NO_SKINNED"});
			auto *mesh_shader = renderer->RequireShader(path, "MSmain", RHIShaderStage::Mesh, {has_skinned ? "HAS_SKINNED" : "NO_SKINNED"});
			auto *frag_shader = renderer->RequireShader(path, "PSmain", RHIShaderStage::Fragment, {has_skinned ? "HAS_SKINNED" : "NO_SKINNED"});

			pipeline_desc.pipeline->SetShader(RHIShaderStage::Task, task_shader);
			pipeline_desc.pipeline->SetShader(RHIShaderStage::Mesh, mesh_shader);
			pipeline_desc.pipeline->SetShader(RHIShaderStage::Fragment, frag_shader);

			pipeline_desc.meta = renderer->RequireShaderMeta(task_shader);
			pipeline_desc.meta += renderer->RequireShaderMeta(mesh_shader);
			pipeline_desc.meta += renderer->RequireShaderMeta(frag_shader);
		}
		else
		{
			auto *vertex_shader = renderer->RequireShader(path, "VSmain", RHIShaderStage::Vertex, {has_skinned ? "HAS_SKINNED" : "NO_SKINNED"});
			auto *frag_shader   = renderer->RequireShader(path, "FSmain", RHIShaderStage::Fragment, {has_skinned ? "HAS_SKINNED" : "NO_SKINNED"});

			pipeline_desc.pipeline->SetShader(RHIShaderStage::Vertex, vertex_shader);
			pipeline_desc.pipeline->SetShader(RHIShaderStage::Fragment, frag_shader);

			if (has_skinned)
			{
				VertexInputState vertex_input_state = {};
				vertex_input_state.input_bindings   = {
                    VertexInputState::InputBinding{0, sizeof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex), RHIVertexInputRate::Vertex}};
				vertex_input_state.input_attributes = {
				    VertexInputState::InputAttribute{RHIVertexSemantics::Position, 0, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, position)},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Texcoord, 3, 0, RHIFormat::R32G32_FLOAT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, texcoord0)},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Blend_Indices, 5, 0, RHIFormat::R32G32B32A32_SINT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, bones[0])},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Blend_Indices, 6, 0, RHIFormat::R32G32B32A32_SINT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, bones[4])},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Blend_Weights, 7, 0, RHIFormat::R32G32B32A32_FLOAT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, weights[0])},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Blend_Weights, 8, 0, RHIFormat::R32G32B32A32_FLOAT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, weights[4])},
				};

				pipeline_desc.pipeline->SetVertexInputState(vertex_input_state);
			}
			else
			{
				VertexInputState vertex_input_state = {};
				vertex_input_state.input_bindings   = {
                    VertexInputState::InputBinding{0, sizeof(Resource<ResourceType::Mesh>::Vertex), RHIVertexInputRate::Vertex}};
				vertex_input_state.input_attributes = {
				    VertexInputState::InputAttribute{RHIVertexSemantics::Position, 0, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, position)},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Texcoord, 3, 0, RHIFormat::R32G32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, texcoord0)},
				};

				pipeline_desc.pipeline->SetVertexInputState(vertex_input_state);
			}

			pipeline_desc.meta = renderer->RequireShaderMeta(vertex_shader);
			pipeline_desc.meta += renderer->RequireShaderMeta(frag_shader);
		}

		return pipeline_desc;
	}

	void RenderShadowMap(
	    RHICommand         *cmd_buffer,
	    const PipelineDesc &mesh_pipeline,
	    const PipelineDesc &skinned_mesh_pipeline,
	    Renderer           *renderer,
	    Scene              *scene,
	    GPUScene           *gpu_scene,
	    Config             *config,
	    ShadowMapData      *shadow_map_data,
	    RHIRenderTarget    *render_target)
	{
		auto    *rhi_context = renderer->GetRHIContext();
		auto     spot_lights = scene->GetComponents<Cmpt::SpotLight>();
		uint32_t layers      = glm::max(static_cast<uint32_t>(spot_lights.size()), 1u);

		RasterizationState rasterization_state;
		rasterization_state.cull_mode         = RHICullMode::None;
		rasterization_state.front_face        = RHIFrontFace::Clockwise;
		rasterization_state.depth_bias_enable = true;
		rasterization_state.depth_bias        = config->bias;
		rasterization_state.depth_bias_slope  = config->slope;
		mesh_pipeline.pipeline->SetRasterizationState(rasterization_state);
		skinned_mesh_pipeline.pipeline->SetRasterizationState(rasterization_state);

		uint32_t size = shadow_map_resolution.at(config->shadow_map_resolution);

		if (!shadow_map_data->shadow_map ||
		    shadow_map_data->shadow_map->GetDesc().width != size ||
		    shadow_map_data->shadow_map->GetDesc().layers < spot_lights.size())
		{
			gpu_scene->light.has_spot_light_shadow = false;
			shadow_map_data->shadow_map            = rhi_context->CreateTexture2DArray(size, size, layers, RHIFormat::D32_FLOAT, RHITextureUsage::RenderTarget | RHITextureUsage::ShaderResource, false);
			cmd_buffer->ResourceStateTransition({TextureStateTransition{
			                                        shadow_map_data->shadow_map.get(),
			                                        RHIResourceState::Undefined,
			                                        RHIResourceState::DepthWrite,
			                                        TextureRange{RHITextureDimension::Texture2DArray, 0, 1, 0, layers}}},
			                                    {});
		}
		else
		{
			cmd_buffer->ResourceStateTransition({TextureStateTransition{
			                                        shadow_map_data->shadow_map.get(),
			                                        RHIResourceState::ShaderResource,
			                                        RHIResourceState::DepthWrite,
			                                        TextureRange{RHITextureDimension::Texture2DArray, 0, 1, 0, layers}}},
			                                    {});
		}

		gpu_scene->light.has_spot_light_shadow = true;

		render_target->Clear();
		render_target->Set(shadow_map_data->shadow_map.get(), RHITextureDimension::Texture2DArray, DepthStencilAttachment{});

		if (rhi_context->IsFeatureSupport(RHIFeature::MeshShading))
		{
			cmd_buffer->BeginMarker("Spot Light Shadow Mapping [Mesh Shader]");

			cmd_buffer->SetViewport(static_cast<float>(render_target->GetWidth()), static_cast<float>(render_target->GetHeight()));
			cmd_buffer->SetScissor(render_target->GetWidth(), render_target->GetHeight());

			cmd_buffer->BeginRenderPass(render_target);

			// Draw Mesh
			if (gpu_scene->mesh_buffer.instance_count > 0)
			{
				auto *descriptor = rhi_context->CreateDescriptor(mesh_pipeline.meta);
				descriptor->BindBuffer("InstanceBuffer", gpu_scene->mesh_buffer.instances.get())
				    .BindBuffer("VertexBuffer", gpu_scene->mesh_buffer.vertex_buffers)
				    .BindBuffer("IndexBuffer", gpu_scene->mesh_buffer.index_buffers)
				    .BindBuffer("MeshletBuffer", gpu_scene->mesh_buffer.meshlet_buffers)
				    .BindBuffer("MeshletDataBuffer", gpu_scene->mesh_buffer.meshlet_data_buffers)
				    .BindBuffer("SpotLightBuffer", gpu_scene->light.spot_light_buffer.get())
				    .BindBuffer("LightInfoBuffer", gpu_scene->light.light_info_buffer.get());

				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(mesh_pipeline.pipeline.get());
				cmd_buffer->DrawMeshTask(gpu_scene->mesh_buffer.max_meshlet_count, gpu_scene->mesh_buffer.instance_count, static_cast<uint32_t>(spot_lights.size()), 32, 1, 1);
			}

			// Draw Skinned Mesh
			if (gpu_scene->skinned_mesh_buffer.instance_count > 0)
			{
				auto *descriptor = rhi_context->CreateDescriptor(skinned_mesh_pipeline.meta);
				descriptor->BindBuffer("InstanceBuffer", gpu_scene->skinned_mesh_buffer.instances.get())
				    .BindBuffer("BoneMatrices", gpu_scene->animation_buffer.bone_matrics)
				    .BindBuffer("VertexBuffer", gpu_scene->skinned_mesh_buffer.vertex_buffers)
				    .BindBuffer("IndexBuffer", gpu_scene->skinned_mesh_buffer.index_buffers)
				    .BindBuffer("MeshletBuffer", gpu_scene->skinned_mesh_buffer.meshlet_buffers)
				    .BindBuffer("MeshletDataBuffer", gpu_scene->skinned_mesh_buffer.meshlet_data_buffers)
				    .BindBuffer("SpotLightBuffer", gpu_scene->light.spot_light_buffer.get())
				    .BindBuffer("LightInfoBuffer", gpu_scene->light.light_info_buffer.get());

				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(skinned_mesh_pipeline.pipeline.get());
				cmd_buffer->DrawMeshTask(gpu_scene->skinned_mesh_buffer.max_meshlet_count, gpu_scene->skinned_mesh_buffer.instance_count, static_cast<uint32_t>(spot_lights.size()), 32, 1, 1);
			}

			cmd_buffer->EndRenderPass();
			cmd_buffer->EndMarker();
		}
		else
		{
			// cmd_buffer->BeginMarker("Spot Light Shadow Mapping");

			// cmd_buffer->SetViewport(static_cast<float>(render_target->GetWidth()), static_cast<float>(render_target->GetHeight()));
			// cmd_buffer->SetScissor(render_target->GetWidth(), render_target->GetHeight());
			// cmd_buffer->BeginRenderPass(render_target);

			//// Draw Mesh
			// if (gpu_scene->mesh_buffer.instance_count > 0)
			//{
			//	auto meshes = renderer->GetScene()->GetComponents<Cmpt::MeshRenderer>();

			//	auto *descriptor = rhi_context->CreateDescriptor(mesh_pipeline.meta);
			//	descriptor->BindBuffer("InstanceBuffer", gpu_scene->mesh_buffer.instances.get());

			//	cmd_buffer->BindDescriptor(descriptor);
			//	cmd_buffer->BindPipelineState(mesh_pipeline.pipeline.get());

			//	uint32_t instance_id = 0;
			//	for (auto &mesh : meshes)
			//	{
			//		auto &submeshes = mesh->GetSubmeshes();
			//		for (auto &submesh : submeshes)
			//		{
			//			auto *resource = renderer->GetResourceManager()->Get<ResourceType::Mesh>(submesh);
			//			if (resource)
			//			{
			//				cmd_buffer->BindVertexBuffer(0, resource->GetVertexBuffer());
			//				cmd_buffer->BindIndexBuffer(resource->GetIndexBuffer());
			//				cmd_buffer->DrawIndexed(static_cast<uint32_t>(resource->GetIndexCount()), 1, 0, 0, instance_id);
			//				instance_id++;
			//			}
			//		}
			//	}
			//}

			//// Draw Skinned Mesh
			// if (gpu_scene->skinned_mesh_buffer.instance_count > 0)
			//{
			//	auto skinned_meshes = renderer->GetScene()->GetComponents<Cmpt::SkinnedMeshRenderer>();

			//	auto *descriptor = rhi_context->CreateDescriptor(skinned_mesh_pipeline.meta);
			//	descriptor->BindBuffer("InstanceBuffer", gpu_scene->skinned_mesh_buffer.instances.get())
			//	    .BindBuffer("BoneMatrices", gpu_scene->animation_buffer.bone_matrics);

			//	cmd_buffer->BindDescriptor(descriptor);
			//	cmd_buffer->BindPipelineState(skinned_mesh_pipeline.pipeline.get());

			//	uint32_t instance_id = 0;
			//	for (auto &skinned_mesh : skinned_meshes)
			//	{
			//		auto &submeshes = skinned_mesh->GetSubmeshes();
			//		for (auto &submesh : submeshes)
			//		{
			//			auto *resource = renderer->GetResourceManager()->Get<ResourceType::SkinnedMesh>(submesh);
			//			if (resource)
			//			{
			//				cmd_buffer->BindVertexBuffer(0, resource->GetVertexBuffer());
			//				cmd_buffer->BindIndexBuffer(resource->GetIndexBuffer());
			//				cmd_buffer->DrawIndexed(static_cast<uint32_t>(resource->GetIndexCount()), 1, 0, 0, instance_id);
			//				instance_id++;
			//			}
			//		}
			//	}
			//}

			// cmd_buffer->EndRenderPass();
			// cmd_buffer->EndMarker();
		}
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                        shadow_map_data->shadow_map.get(),
		                                        RHIResourceState::DepthWrite,
		                                        RHIResourceState::ShaderResource,
		                                        TextureRange{RHITextureDimension::Texture2DArray, 0, 1, 0, layers}}},
		                                    {});
	}

	void RenderCascadeShadowMap(
	    RHICommand         *cmd_buffer,
	    const PipelineDesc &mesh_pipeline,
	    const PipelineDesc &skinned_mesh_pipeline,
	    Renderer           *renderer,
	    Scene              *scene,
	    GPUScene           *gpu_scene,
	    Config             *config,
	    ShadowMapData      *shadow_map_data,
	    RHIRenderTarget    *render_target)
	{
		auto    *rhi_context        = renderer->GetRHIContext();
		auto     directional_lights = scene->GetComponents<Cmpt::DirectionalLight>();
		uint32_t layers             = glm::max(static_cast<uint32_t>(directional_lights.size() * 4ull), 1u);

		RasterizationState rasterization_state;
		rasterization_state.cull_mode         = RHICullMode::None;
		rasterization_state.front_face        = RHIFrontFace::Clockwise;
		rasterization_state.depth_bias_enable = true;
		rasterization_state.depth_bias        = config->bias;
		rasterization_state.depth_bias_slope  = config->slope;
		mesh_pipeline.pipeline->SetRasterizationState(rasterization_state);
		skinned_mesh_pipeline.pipeline->SetRasterizationState(rasterization_state);

		uint32_t size = shadow_map_resolution.at(config->cascade_shadow_map_resolution);

		if (!shadow_map_data->cascade_shadow_map ||
		    shadow_map_data->cascade_shadow_map->GetDesc().width != size ||
		    shadow_map_data->cascade_shadow_map->GetDesc().layers < directional_lights.size() * 4)
		{
			gpu_scene->light.has_directional_light_shadow = false;
			shadow_map_data->cascade_shadow_map           = rhi_context->CreateTexture2DArray(size, size, layers, RHIFormat::D32_FLOAT, RHITextureUsage::RenderTarget | RHITextureUsage::ShaderResource, false);
			cmd_buffer->ResourceStateTransition({TextureStateTransition{
			                                        shadow_map_data->cascade_shadow_map.get(),
			                                        RHIResourceState::Undefined,
			                                        RHIResourceState::DepthWrite,
			                                        TextureRange{RHITextureDimension::Texture2DArray, 0, 1, 0, layers}}},
			                                    {});
		}
		else
		{
			cmd_buffer->ResourceStateTransition({TextureStateTransition{
			                                        shadow_map_data->cascade_shadow_map.get(),
			                                        RHIResourceState::ShaderResource,
			                                        RHIResourceState::DepthWrite,
			                                        TextureRange{RHITextureDimension::Texture2DArray, 0, 1, 0, layers}}},
			                                    {});
		}

		gpu_scene->light.has_directional_light_shadow = true;

		render_target->Clear();
		render_target->Set(shadow_map_data->cascade_shadow_map.get(), RHITextureDimension::Texture2DArray, DepthStencilAttachment{});

		if (rhi_context->IsFeatureSupport(RHIFeature::MeshShading))
		{
			cmd_buffer->BeginMarker("Directional Light Shadow Mapping [Mesh Shader]");

			cmd_buffer->SetViewport(static_cast<float>(render_target->GetWidth()), static_cast<float>(render_target->GetHeight()));
			cmd_buffer->SetScissor(render_target->GetWidth(), render_target->GetHeight());

			cmd_buffer->BeginRenderPass(render_target);

			// Draw Mesh
			if (gpu_scene->mesh_buffer.instance_count > 0)
			{
				auto *descriptor = rhi_context->CreateDescriptor(mesh_pipeline.meta);
				descriptor->BindBuffer("InstanceBuffer", gpu_scene->mesh_buffer.instances.get())
				    .BindBuffer("VertexBuffer", gpu_scene->mesh_buffer.vertex_buffers)
				    .BindBuffer("IndexBuffer", gpu_scene->mesh_buffer.index_buffers)
				    .BindBuffer("MeshletBuffer", gpu_scene->mesh_buffer.meshlet_buffers)
				    .BindBuffer("MeshletDataBuffer", gpu_scene->mesh_buffer.meshlet_data_buffers)
				    .BindBuffer("DirectionalLightBuffer", gpu_scene->light.directional_light_buffer.get())
				    .BindBuffer("LightInfoBuffer", gpu_scene->light.light_info_buffer.get());

				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(mesh_pipeline.pipeline.get());
				cmd_buffer->DrawMeshTask(gpu_scene->mesh_buffer.max_meshlet_count, gpu_scene->mesh_buffer.instance_count, static_cast<uint32_t>(directional_lights.size() * 4), 32, 1, 1);
			}

			// Draw Skinned Mesh
			if (gpu_scene->skinned_mesh_buffer.instance_count > 0)
			{
				auto *descriptor = rhi_context->CreateDescriptor(skinned_mesh_pipeline.meta);
				descriptor->BindBuffer("InstanceBuffer", gpu_scene->skinned_mesh_buffer.instances.get())
				    .BindBuffer("BoneMatrices", gpu_scene->animation_buffer.bone_matrics)
				    .BindBuffer("VertexBuffer", gpu_scene->skinned_mesh_buffer.vertex_buffers)
				    .BindBuffer("IndexBuffer", gpu_scene->skinned_mesh_buffer.index_buffers)
				    .BindBuffer("MeshletBuffer", gpu_scene->skinned_mesh_buffer.meshlet_buffers)
				    .BindBuffer("MeshletDataBuffer", gpu_scene->skinned_mesh_buffer.meshlet_data_buffers)
				    .BindBuffer("DirectionalLightBuffer", gpu_scene->light.directional_light_buffer.get())
				    .BindBuffer("LightInfoBuffer", gpu_scene->light.light_info_buffer.get());

				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(skinned_mesh_pipeline.pipeline.get());
				cmd_buffer->DrawMeshTask(gpu_scene->skinned_mesh_buffer.max_meshlet_count, gpu_scene->skinned_mesh_buffer.instance_count, static_cast<uint32_t>(directional_lights.size() * 4), 32, 1, 1);
			}

			cmd_buffer->EndRenderPass();
			cmd_buffer->EndMarker();
		}
		else
		{
			// cmd_buffer->BeginMarker("Spot Light Shadow Mapping");

			// cmd_buffer->SetViewport(static_cast<float>(render_target->GetWidth()), static_cast<float>(render_target->GetHeight()));
			// cmd_buffer->SetScissor(render_target->GetWidth(), render_target->GetHeight());
			// cmd_buffer->BeginRenderPass(render_target);

			//// Draw Mesh
			// if (gpu_scene->mesh_buffer.instance_count > 0)
			//{
			//	auto meshes = renderer->GetScene()->GetComponents<Cmpt::MeshRenderer>();

			//	auto *descriptor = rhi_context->CreateDescriptor(mesh_pipeline.meta);
			//	descriptor->BindBuffer("InstanceBuffer", gpu_scene->mesh_buffer.instances.get());

			//	cmd_buffer->BindDescriptor(descriptor);
			//	cmd_buffer->BindPipelineState(mesh_pipeline.pipeline.get());

			//	uint32_t instance_id = 0;
			//	for (auto &mesh : meshes)
			//	{
			//		auto &submeshes = mesh->GetSubmeshes();
			//		for (auto &submesh : submeshes)
			//		{
			//			auto *resource = renderer->GetResourceManager()->Get<ResourceType::Mesh>(submesh);
			//			if (resource)
			//			{
			//				cmd_buffer->BindVertexBuffer(0, resource->GetVertexBuffer());
			//				cmd_buffer->BindIndexBuffer(resource->GetIndexBuffer());
			//				cmd_buffer->DrawIndexed(static_cast<uint32_t>(resource->GetIndexCount()), 1, 0, 0, instance_id);
			//				instance_id++;
			//			}
			//		}
			//	}
			//}

			//// Draw Skinned Mesh
			// if (gpu_scene->skinned_mesh_buffer.instance_count > 0)
			//{
			//	auto skinned_meshes = renderer->GetScene()->GetComponents<Cmpt::SkinnedMeshRenderer>();

			//	auto *descriptor = rhi_context->CreateDescriptor(skinned_mesh_pipeline.meta);
			//	descriptor->BindBuffer("InstanceBuffer", gpu_scene->skinned_mesh_buffer.instances.get())
			//	    .BindBuffer("BoneMatrices", gpu_scene->animation_buffer.bone_matrics);

			//	cmd_buffer->BindDescriptor(descriptor);
			//	cmd_buffer->BindPipelineState(skinned_mesh_pipeline.pipeline.get());

			//	uint32_t instance_id = 0;
			//	for (auto &skinned_mesh : skinned_meshes)
			//	{
			//		auto &submeshes = skinned_mesh->GetSubmeshes();
			//		for (auto &submesh : submeshes)
			//		{
			//			auto *resource = renderer->GetResourceManager()->Get<ResourceType::SkinnedMesh>(submesh);
			//			if (resource)
			//			{
			//				cmd_buffer->BindVertexBuffer(0, resource->GetVertexBuffer());
			//				cmd_buffer->BindIndexBuffer(resource->GetIndexBuffer());
			//				cmd_buffer->DrawIndexed(static_cast<uint32_t>(resource->GetIndexCount()), 1, 0, 0, instance_id);
			//				instance_id++;
			//			}
			//		}
			//	}
			//}

			// cmd_buffer->EndRenderPass();
			// cmd_buffer->EndMarker();
		}
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                        shadow_map_data->cascade_shadow_map.get(),
		                                        RHIResourceState::DepthWrite,
		                                        RHIResourceState::ShaderResource,
		                                        TextureRange{RHITextureDimension::Texture2DArray, 0, 1, 0, layers}}},
		                                    {});
	}

	void RenderOmniShadowMap(
	    RHICommand         *cmd_buffer,
	    const PipelineDesc &mesh_pipeline,
	    const PipelineDesc &skinned_mesh_pipeline,
	    Renderer           *renderer,
	    Scene              *scene,
	    GPUScene           *gpu_scene,
	    Config             *config,
	    ShadowMapData      *shadow_map_data,
	    RHIRenderTarget    *render_target)
	{
		auto    *rhi_context  = renderer->GetRHIContext();
		auto     point_lights = scene->GetComponents<Cmpt::PointLight>();
		uint32_t layers       = glm::max(static_cast<uint32_t>(point_lights.size() * 6ull), 1u);

		uint32_t size = shadow_map_resolution.at(config->omni_map_resolution);

		if (!shadow_map_data->omni_shadow_map ||
		    shadow_map_data->omni_shadow_map->GetDesc().width != size ||
		    shadow_map_data->omni_shadow_map->GetDesc().layers < point_lights.size() * 6)
		{
			gpu_scene->light.has_point_light_shadow = false;
			shadow_map_data->omni_shadow_map        = rhi_context->CreateTexture2DArray(size, size, layers, RHIFormat::D32_FLOAT, RHITextureUsage::RenderTarget | RHITextureUsage::ShaderResource, false);
			cmd_buffer->ResourceStateTransition({TextureStateTransition{
			                                        shadow_map_data->omni_shadow_map.get(),
			                                        RHIResourceState::Undefined,
			                                        RHIResourceState::DepthWrite,
			                                        TextureRange{RHITextureDimension::Texture2DArray, 0, 1, 0, layers}}},
			                                    {});
		}
		else
		{
			cmd_buffer->ResourceStateTransition({TextureStateTransition{
			                                        shadow_map_data->omni_shadow_map.get(),
			                                        RHIResourceState::ShaderResource,
			                                        RHIResourceState::DepthWrite,
			                                        TextureRange{RHITextureDimension::Texture2DArray, 0, 1, 0, layers}}},
			                                    {});
		}

		gpu_scene->light.has_point_light_shadow = true;

		render_target->Clear();
		render_target->Set(shadow_map_data->omni_shadow_map.get(), RHITextureDimension::Texture2DArray, DepthStencilAttachment{});

		if (rhi_context->IsFeatureSupport(RHIFeature::MeshShading))
		{
			cmd_buffer->BeginMarker("Point Light Shadow Mapping [Mesh Shader]");

			cmd_buffer->SetViewport(static_cast<float>(render_target->GetWidth()), static_cast<float>(render_target->GetHeight()));
			cmd_buffer->SetScissor(render_target->GetWidth(), render_target->GetHeight());

			cmd_buffer->BeginRenderPass(render_target);

			// Draw Mesh
			if (gpu_scene->mesh_buffer.instance_count > 0)
			{
				auto *descriptor = rhi_context->CreateDescriptor(mesh_pipeline.meta);
				descriptor->BindBuffer("InstanceBuffer", gpu_scene->mesh_buffer.instances.get())
				    .BindBuffer("VertexBuffer", gpu_scene->mesh_buffer.vertex_buffers)
				    .BindBuffer("IndexBuffer", gpu_scene->mesh_buffer.index_buffers)
				    .BindBuffer("MeshletBuffer", gpu_scene->mesh_buffer.meshlet_buffers)
				    .BindBuffer("MeshletDataBuffer", gpu_scene->mesh_buffer.meshlet_data_buffers)
				    .BindBuffer("PointLightBuffer", gpu_scene->light.point_light_buffer.get())
				    .BindBuffer("LightInfoBuffer", gpu_scene->light.light_info_buffer.get());

				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(mesh_pipeline.pipeline.get());
				cmd_buffer->DrawMeshTask(gpu_scene->mesh_buffer.max_meshlet_count, gpu_scene->mesh_buffer.instance_count, static_cast<uint32_t>(point_lights.size() * 6), 32, 1, 1);
			}

			// Draw Skinned Mesh
			if (gpu_scene->skinned_mesh_buffer.instance_count > 0)
			{
				auto *descriptor = rhi_context->CreateDescriptor(skinned_mesh_pipeline.meta);
				descriptor->BindBuffer("InstanceBuffer", gpu_scene->skinned_mesh_buffer.instances.get())
				    .BindBuffer("BoneMatrices", gpu_scene->animation_buffer.bone_matrics)
				    .BindBuffer("VertexBuffer", gpu_scene->skinned_mesh_buffer.vertex_buffers)
				    .BindBuffer("IndexBuffer", gpu_scene->skinned_mesh_buffer.index_buffers)
				    .BindBuffer("MeshletBuffer", gpu_scene->skinned_mesh_buffer.meshlet_buffers)
				    .BindBuffer("MeshletDataBuffer", gpu_scene->skinned_mesh_buffer.meshlet_data_buffers)
				    .BindBuffer("PointLightBuffer", gpu_scene->light.point_light_buffer.get())
				    .BindBuffer("LightInfoBuffer", gpu_scene->light.light_info_buffer.get());

				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(skinned_mesh_pipeline.pipeline.get());
				cmd_buffer->DrawMeshTask(gpu_scene->skinned_mesh_buffer.max_meshlet_count, gpu_scene->skinned_mesh_buffer.instance_count, static_cast<uint32_t>(point_lights.size() * 6), 32, 1, 1);
			}

			cmd_buffer->EndRenderPass();
			cmd_buffer->EndMarker();
		}
		else
		{
			// cmd_buffer->BeginMarker("Spot Light Shadow Mapping");

			// cmd_buffer->SetViewport(static_cast<float>(render_target->GetWidth()), static_cast<float>(render_target->GetHeight()));
			// cmd_buffer->SetScissor(render_target->GetWidth(), render_target->GetHeight());
			// cmd_buffer->BeginRenderPass(render_target);

			//// Draw Mesh
			// if (gpu_scene->mesh_buffer.instance_count > 0)
			//{
			//	auto meshes = renderer->GetScene()->GetComponents<Cmpt::MeshRenderer>();

			//	auto *descriptor = rhi_context->CreateDescriptor(mesh_pipeline.meta);
			//	descriptor->BindBuffer("InstanceBuffer", gpu_scene->mesh_buffer.instances.get());

			//	cmd_buffer->BindDescriptor(descriptor);
			//	cmd_buffer->BindPipelineState(mesh_pipeline.pipeline.get());

			//	uint32_t instance_id = 0;
			//	for (auto &mesh : meshes)
			//	{
			//		auto &submeshes = mesh->GetSubmeshes();
			//		for (auto &submesh : submeshes)
			//		{
			//			auto *resource = renderer->GetResourceManager()->Get<ResourceType::Mesh>(submesh);
			//			if (resource)
			//			{
			//				cmd_buffer->BindVertexBuffer(0, resource->GetVertexBuffer());
			//				cmd_buffer->BindIndexBuffer(resource->GetIndexBuffer());
			//				cmd_buffer->DrawIndexed(static_cast<uint32_t>(resource->GetIndexCount()), 1, 0, 0, instance_id);
			//				instance_id++;
			//			}
			//		}
			//	}
			//}

			//// Draw Skinned Mesh
			// if (gpu_scene->skinned_mesh_buffer.instance_count > 0)
			//{
			//	auto skinned_meshes = renderer->GetScene()->GetComponents<Cmpt::SkinnedMeshRenderer>();

			//	auto *descriptor = rhi_context->CreateDescriptor(skinned_mesh_pipeline.meta);
			//	descriptor->BindBuffer("InstanceBuffer", gpu_scene->skinned_mesh_buffer.instances.get())
			//	    .BindBuffer("BoneMatrices", gpu_scene->animation_buffer.bone_matrics);

			//	cmd_buffer->BindDescriptor(descriptor);
			//	cmd_buffer->BindPipelineState(skinned_mesh_pipeline.pipeline.get());

			//	uint32_t instance_id = 0;
			//	for (auto &skinned_mesh : skinned_meshes)
			//	{
			//		auto &submeshes = skinned_mesh->GetSubmeshes();
			//		for (auto &submesh : submeshes)
			//		{
			//			auto *resource = renderer->GetResourceManager()->Get<ResourceType::SkinnedMesh>(submesh);
			//			if (resource)
			//			{
			//				cmd_buffer->BindVertexBuffer(0, resource->GetVertexBuffer());
			//				cmd_buffer->BindIndexBuffer(resource->GetIndexBuffer());
			//				cmd_buffer->DrawIndexed(static_cast<uint32_t>(resource->GetIndexCount()), 1, 0, 0, instance_id);
			//				instance_id++;
			//			}
			//		}
			//	}
			//}

			// cmd_buffer->EndRenderPass();
			// cmd_buffer->EndMarker();
		}
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                        shadow_map_data->omni_shadow_map.get(),
		                                        RHIResourceState::DepthWrite,
		                                        RHIResourceState::ShaderResource,
		                                        TextureRange{RHITextureDimension::Texture2DArray, 0, 1, 0, layers}}},
		                                    {});
	}

	virtual void OnImGui(Variant *config)
	{
		Config *config_data = config->Convert<Config>();

		ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x / 2.f);
		ImGui::DragFloat("Depth Bias", &config_data->bias, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
		ImGui::DragFloat("Depth Slope", &config_data->slope, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");

		const char *resolution[] = {"VeryLow", "Low", "Medium", "High", "VeryHigh"};
		ImGui::Text("Shadow Map Resolution");
		ImGui::Combo("Spot Light", reinterpret_cast<int32_t *>(&config_data->shadow_map_resolution), resolution, 5);
		ImGui::Combo("Point Light", reinterpret_cast<int32_t *>(&config_data->omni_map_resolution), resolution, 5);
		ImGui::Combo("Directional Light", reinterpret_cast<int32_t *>(&config_data->cascade_shadow_map_resolution), resolution, 5);

		ImGui::PopItemWidth();
	}
};

CONFIGURATION_PASS(ShadowMapPass)