#include "IPass.hpp"

#include <Resource/Resource/Mesh.hpp>
#include <Resource/Resource/SkinnedMesh.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Components/AllComponents.hpp>
#include <Scene/Scene.hpp>

using namespace Ilum;

class VisibilityGeometryPass : public RenderPass<VisibilityGeometryPass>
{
	struct PipelineDesc
	{
		std::shared_ptr<RHIPipelineState> pipeline = nullptr;
		ShaderMeta                        meta;
	};

  public:
	VisibilityGeometryPass() = default;

	~VisibilityGeometryPass() = default;

	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::Rasterization)
		    .SetName("VisibilityGeometryPass")
		    .SetCategory("RenderPath")
		    .WriteTexture2D(handle++, "Visibility Buffer", 0, 0, RHIFormat::R32_UINT, RHIResourceState::RenderTarget)
		    .WriteTexture2D(handle++, "Depth Buffer", 0, 0, RHIFormat::D32_FLOAT, RHIResourceState::DepthWrite);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		std::shared_ptr<RHIRenderTarget> render_target = std::shared_ptr<RHIRenderTarget>(std::move(renderer->GetRHIContext()->CreateRenderTarget()));

		auto mesh_pipeline = CreatePipeline(renderer, false);
		auto skinned_mesh_pipeline = CreatePipeline(renderer, true);

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto  visibility_buffer = render_graph.GetTexture(desc.GetPin("Visibility Buffer").handle);
			auto  depth_stencil     = render_graph.GetTexture(desc.GetPin("Depth Buffer").handle);
			auto *rhi_context       = renderer->GetRHIContext();
			auto *gpu_scene         = black_board.Get<GPUScene>();
			auto *view              = black_board.Get<View>();

			render_target->Clear();
			render_target->Set(0, visibility_buffer, TextureRange{}, ColorAttachment{});
			render_target->Set(depth_stencil, TextureRange{}, DepthStencilAttachment{});

			// Mesh Shading
			if (rhi_context->IsFeatureSupport(RHIFeature::MeshShading))
			{
				cmd_buffer->SetViewport(static_cast<float>(render_target->GetWidth()), static_cast<float>(render_target->GetHeight()));
				cmd_buffer->SetScissor(render_target->GetWidth(), render_target->GetHeight());
				cmd_buffer->BeginRenderPass(render_target.get());

				// Draw Mesh
				if (gpu_scene->mesh_buffer.instance_count > 0)
				{
					auto *descriptor = rhi_context->CreateDescriptor(mesh_pipeline.meta);
					descriptor->BindBuffer("InstanceBuffer", gpu_scene->mesh_buffer.instances.get())
					    .BindBuffer("ViewBuffer", view->buffer.get())
					    .BindBuffer("VertexBuffer", gpu_scene->mesh_buffer.vertex_buffers)
					    .BindBuffer("IndexBuffer", gpu_scene->mesh_buffer.index_buffers)
					    .BindBuffer("MeshletBuffer", gpu_scene->mesh_buffer.meshlet_buffers)
					    .BindBuffer("MeshletDataBuffer", gpu_scene->mesh_buffer.meshlet_data_buffers);

					cmd_buffer->BindDescriptor(descriptor);
					cmd_buffer->BindPipelineState(mesh_pipeline.pipeline.get());
					cmd_buffer->DrawMeshTask(gpu_scene->mesh_buffer.max_meshlet_count, gpu_scene->mesh_buffer.instance_count, 1, 32, 1, 1);
				}

				// Draw Skinned Mesh
				if (gpu_scene->skinned_mesh_buffer.instance_count > 0)
				{
					auto *descriptor = rhi_context->CreateDescriptor(skinned_mesh_pipeline.meta);
					descriptor->BindBuffer("InstanceBuffer", gpu_scene->skinned_mesh_buffer.instances.get())
					    .BindBuffer("ViewBuffer", view->buffer.get())
					    .BindBuffer("BoneMatrices", gpu_scene->animation_buffer.bone_matrics)
					    .BindBuffer("VertexBuffer", gpu_scene->skinned_mesh_buffer.vertex_buffers)
					    .BindBuffer("IndexBuffer", gpu_scene->skinned_mesh_buffer.index_buffers)
					    .BindBuffer("MeshletBuffer", gpu_scene->skinned_mesh_buffer.meshlet_buffers)
					    .BindBuffer("MeshletDataBuffer", gpu_scene->skinned_mesh_buffer.meshlet_data_buffers);

					cmd_buffer->BindDescriptor(descriptor);
					cmd_buffer->BindPipelineState(skinned_mesh_pipeline.pipeline.get());
					cmd_buffer->DrawMeshTask(gpu_scene->skinned_mesh_buffer.max_meshlet_count, gpu_scene->skinned_mesh_buffer.instance_count, 1, 32, 1, 1);
				}

				cmd_buffer->EndRenderPass();
			}
			else
			{
				cmd_buffer->SetViewport(static_cast<float>(render_target->GetWidth()), static_cast<float>(render_target->GetHeight()));
				cmd_buffer->SetScissor(render_target->GetWidth(), render_target->GetHeight());
				cmd_buffer->BeginRenderPass(render_target.get());

				// Draw Mesh
				if (gpu_scene->mesh_buffer.instance_count > 0)
				{
					auto meshes = renderer->GetScene()->GetComponents<Cmpt::MeshRenderer>();

					auto *descriptor = rhi_context->CreateDescriptor(mesh_pipeline.meta);
					descriptor->BindBuffer("InstanceBuffer", gpu_scene->mesh_buffer.instances.get())
					    .BindBuffer("ViewBuffer", view->buffer.get());

					cmd_buffer->BindDescriptor(descriptor);
					cmd_buffer->BindPipelineState(mesh_pipeline.pipeline.get());

					uint32_t instance_id = 0;
					for (auto &mesh : meshes)
					{
						auto &submeshes = mesh->GetSubmeshes();
						for (auto &submesh : submeshes)
						{
							auto *resource = renderer->GetResourceManager()->Get<ResourceType::Mesh>(submesh);
							if (resource)
							{
								cmd_buffer->BindVertexBuffer(0, resource->GetVertexBuffer());
								cmd_buffer->BindIndexBuffer(resource->GetIndexBuffer());
								cmd_buffer->DrawIndexed(static_cast<uint32_t>(resource->GetIndexCount()), 1, 0, 0, instance_id);
								instance_id++;
							}
						}
					}
				}

				// Draw Skinned Mesh
				if (gpu_scene->skinned_mesh_buffer.instance_count > 0)
				{
					auto skinned_meshes = renderer->GetScene()->GetComponents<Cmpt::SkinnedMeshRenderer>();

					auto *descriptor = rhi_context->CreateDescriptor(skinned_mesh_pipeline.meta);
					descriptor->BindBuffer("InstanceBuffer", gpu_scene->skinned_mesh_buffer.instances.get())
					    .BindBuffer("ViewBuffer", view->buffer.get())
					    .BindBuffer("BoneMatrices", gpu_scene->animation_buffer.bone_matrics);

					cmd_buffer->BindDescriptor(descriptor);
					cmd_buffer->BindPipelineState(skinned_mesh_pipeline.pipeline.get());

					uint32_t instance_id = 0;
					for (auto &skinned_mesh : skinned_meshes)
					{
						auto &submeshes = skinned_mesh->GetSubmeshes();
						for (auto &submesh : submeshes)
						{
							auto *resource = renderer->GetResourceManager()->Get<ResourceType::SkinnedMesh>(submesh);
							if (resource)
							{
								cmd_buffer->BindVertexBuffer(0, resource->GetVertexBuffer());
								cmd_buffer->BindIndexBuffer(resource->GetIndexBuffer());
								cmd_buffer->DrawIndexed(static_cast<uint32_t>(resource->GetIndexCount()), 1, 0, 0, instance_id);
								instance_id++;
							}
						}
					}
				}

				cmd_buffer->EndRenderPass();
			}
		};
	}

	virtual void OnImGui(Variant *config)
	{
	}

	PipelineDesc CreatePipeline(Renderer *renderer, bool has_skinned)
	{
		auto *rhi_context = renderer->GetRHIContext();

		PipelineDesc pipeline_desc;
		pipeline_desc.pipeline = std::shared_ptr<RHIPipelineState>(std::move(rhi_context->CreatePipelineState()));

		if (rhi_context->IsFeatureSupport(RHIFeature::MeshShading))
		{
			auto *task_shader = renderer->RequireShader("Source/Shaders/RenderPath/VisibilityGeometryPass.hlsl", "ASmain", RHIShaderStage::Task, {has_skinned ? "HAS_SKINNED" : "NO_SKINNED"});
			auto *mesh_shader = renderer->RequireShader("Source/Shaders/RenderPath/VisibilityGeometryPass.hlsl", "MSmain", RHIShaderStage::Mesh, {has_skinned ? "HAS_SKINNED" : "NO_SKINNED"});
			auto *frag_shader = renderer->RequireShader("Source/Shaders/RenderPath/VisibilityGeometryPass.hlsl", "PSmain", RHIShaderStage::Fragment, {has_skinned ? "HAS_SKINNED" : "NO_SKINNED"});

			pipeline_desc.pipeline->SetShader(RHIShaderStage::Task, task_shader);
			pipeline_desc.pipeline->SetShader(RHIShaderStage::Mesh, mesh_shader);
			pipeline_desc.pipeline->SetShader(RHIShaderStage::Fragment, frag_shader);

			BlendState blend_state;
			blend_state.attachment_states.resize(1);
			pipeline_desc.pipeline->SetBlendState(blend_state);

			RasterizationState rasterization_state;
			rasterization_state.cull_mode  = RHICullMode::None;
			rasterization_state.front_face = RHIFrontFace::Clockwise;
			pipeline_desc.pipeline->SetRasterizationState(rasterization_state);

			DepthStencilState depth_stencil_state  = {};
			depth_stencil_state.depth_write_enable = true;
			depth_stencil_state.depth_test_enable  = true;
			pipeline_desc.pipeline->SetDepthStencilState(depth_stencil_state);

			pipeline_desc.meta += renderer->RequireShaderMeta(task_shader);
			pipeline_desc.meta += renderer->RequireShaderMeta(mesh_shader);
			pipeline_desc.meta += renderer->RequireShaderMeta(frag_shader);
		}
		else
		{
			auto *vertex_shader = renderer->RequireShader("Source/Shaders/RenderPath/VisibilityGeometryPass.hlsl", "VSmain", RHIShaderStage::Vertex, {has_skinned ? "HAS_SKINNED" : "NO_SKINNED"});
			auto *frag_shader   = renderer->RequireShader("Source/Shaders/RenderPath/VisibilityGeometryPass.hlsl", "FSmain", RHIShaderStage::Fragment, {has_skinned ? "HAS_SKINNED" : "NO_SKINNED"});

			pipeline_desc.pipeline->SetShader(RHIShaderStage::Vertex, vertex_shader);
			pipeline_desc.pipeline->SetShader(RHIShaderStage::Fragment, frag_shader);

			BlendState blend_state;
			blend_state.attachment_states.resize(1);
			pipeline_desc.pipeline->SetBlendState(blend_state);

			RasterizationState rasterization_state;
			rasterization_state.cull_mode  = RHICullMode::None;
			rasterization_state.front_face = RHIFrontFace::Clockwise;
			pipeline_desc.pipeline->SetRasterizationState(rasterization_state);

			DepthStencilState depth_stencil_state  = {};
			depth_stencil_state.depth_write_enable = true;
			depth_stencil_state.depth_test_enable  = true;
			pipeline_desc.pipeline->SetDepthStencilState(depth_stencil_state);

			VertexInputState vertex_input_state = {};
			if (has_skinned)
			{
				vertex_input_state.input_bindings = {
				    VertexInputState::InputBinding{0, sizeof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex), RHIVertexInputRate::Vertex}};
				vertex_input_state.input_attributes = {
				    VertexInputState::InputAttribute{RHIVertexSemantics::Position, 0, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, position)},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Texcoord, 3, 0, RHIFormat::R32G32_FLOAT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, texcoord0)},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Blend_Indices, 5, 0, RHIFormat::R32G32B32A32_SINT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, bones[0])},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Blend_Indices, 6, 0, RHIFormat::R32G32B32A32_SINT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, bones[4])},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Blend_Weights, 7, 0, RHIFormat::R32G32B32A32_FLOAT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, weights[0])},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Blend_Weights, 8, 0, RHIFormat::R32G32B32A32_FLOAT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, weights[4])},
				};
			}
			else
			{
				vertex_input_state.input_bindings = {
				    VertexInputState::InputBinding{0, sizeof(Resource<ResourceType::Mesh>::Vertex), RHIVertexInputRate::Vertex}};
				vertex_input_state.input_attributes = {
				    VertexInputState::InputAttribute{RHIVertexSemantics::Position, 0, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, position)},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Texcoord, 3, 0, RHIFormat::R32G32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, texcoord0)},
				};
			}

			pipeline_desc.pipeline->SetVertexInputState(vertex_input_state);
			pipeline_desc.meta += renderer->RequireShaderMeta(vertex_shader);
			pipeline_desc.meta += renderer->RequireShaderMeta(frag_shader);
		}

		return pipeline_desc;
	}
};

CONFIGURATION_PASS(VisibilityGeometryPass)