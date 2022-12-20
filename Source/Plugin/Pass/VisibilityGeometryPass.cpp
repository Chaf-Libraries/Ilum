#include "IPass.hpp"

#include <Components/AllComponents.hpp>
#include <Resource/Resource/Mesh.hpp>
#include <Resource/ResourceManager.hpp>
#include <SceneGraph/Scene.hpp>

using namespace Ilum;

class VisibilityGeometryPass : public IPass<VisibilityGeometryPass>
{
  public:
	VisibilityGeometryPass() = default;

	~VisibilityGeometryPass() = default;

	virtual void CreateDesc(RenderPassDesc *desc)
	{
		desc->SetBindPoint(BindPoint::Rasterization)
		    .Write("Visibility Buffer", RenderResourceDesc::Type::Texture, RHIResourceState::RenderTarget)
		    .Write("Depth Stencil", RenderResourceDesc::Type::Texture, RHIResourceState::DepthWrite);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		struct
		{
			std::shared_ptr<RHIRenderTarget>  render_target = nullptr;
			std::shared_ptr<RHIPipelineState> pipeline      = nullptr;
			ShaderMeta                        meta;
		} mesh_visibility_pipeline;

		mesh_visibility_pipeline.pipeline      = std::shared_ptr<RHIPipelineState>(std::move(renderer->GetRHIContext()->CreatePipelineState()));
		mesh_visibility_pipeline.render_target = std::shared_ptr<RHIRenderTarget>(std::move(renderer->GetRHIContext()->CreateRenderTarget()));

		if (renderer->GetRHIContext()->IsFeatureSupport(RHIFeature::MeshShading))
		{
			auto *task_shader = renderer->RequireShader("Source/Shaders/VisibilityMeshPass.hlsl", "ASmain", RHIShaderStage::Task);
			auto *mesh_shader = renderer->RequireShader("Source/Shaders/VisibilityMeshPass.hlsl", "MSmain", RHIShaderStage::Mesh);
			auto *frag_shader = renderer->RequireShader("Source/Shaders/VisibilityMeshPass.hlsl", "PSmain", RHIShaderStage::Fragment);

			mesh_visibility_pipeline.pipeline->SetShader(RHIShaderStage::Task, task_shader);
			mesh_visibility_pipeline.pipeline->SetShader(RHIShaderStage::Mesh, mesh_shader);
			mesh_visibility_pipeline.pipeline->SetShader(RHIShaderStage::Fragment, frag_shader);

			BlendState blend_state;
			blend_state.attachment_states.resize(1);
			mesh_visibility_pipeline.pipeline->SetBlendState(blend_state);

			RasterizationState rasterization_state;
			rasterization_state.cull_mode  = RHICullMode::None;
			rasterization_state.front_face = RHIFrontFace::Clockwise;
			mesh_visibility_pipeline.pipeline->SetRasterizationState(rasterization_state);

			DepthStencilState depth_stencil_state  = {};
			depth_stencil_state.depth_write_enable = true;
			depth_stencil_state.depth_test_enable  = true;
			mesh_visibility_pipeline.pipeline->SetDepthStencilState(depth_stencil_state);

			mesh_visibility_pipeline.meta = renderer->RequireShaderMeta(task_shader);
			mesh_visibility_pipeline.meta += renderer->RequireShaderMeta(mesh_shader);
			mesh_visibility_pipeline.meta += renderer->RequireShaderMeta(frag_shader);
		}
		else
		{
			auto *vertex_shader = renderer->RequireShader("Source/Shaders/VisibilityMeshPass.hlsl", "VSmain", RHIShaderStage::Vertex);
			auto *frag_shader   = renderer->RequireShader("Source/Shaders/VisibilityMeshPass.hlsl", "FSmain", RHIShaderStage::Fragment);

			mesh_visibility_pipeline.pipeline->SetShader(RHIShaderStage::Vertex, vertex_shader);
			mesh_visibility_pipeline.pipeline->SetShader(RHIShaderStage::Fragment, frag_shader);

			BlendState blend_state;
			blend_state.attachment_states.resize(1);
			mesh_visibility_pipeline.pipeline->SetBlendState(blend_state);

			RasterizationState rasterization_state;
			rasterization_state.cull_mode  = RHICullMode::None;
			rasterization_state.front_face = RHIFrontFace::Clockwise;
			mesh_visibility_pipeline.pipeline->SetRasterizationState(rasterization_state);

			DepthStencilState depth_stencil_state  = {};
			depth_stencil_state.depth_write_enable = true;
			depth_stencil_state.depth_test_enable  = true;
			mesh_visibility_pipeline.pipeline->SetDepthStencilState(depth_stencil_state);

			VertexInputState vertex_input_state = {};
			vertex_input_state.input_bindings   = {
                VertexInputState::InputBinding{0, sizeof(Resource<ResourceType::Mesh>::Vertex), RHIVertexInputRate::Vertex}};
			vertex_input_state.input_attributes = {
			    VertexInputState::InputAttribute{RHIVertexSemantics::Position, 0, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, position)},
			    VertexInputState::InputAttribute{RHIVertexSemantics::Normal, 1, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, normal)},
			    VertexInputState::InputAttribute{RHIVertexSemantics::Tangent, 2, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, tangent)},
			    VertexInputState::InputAttribute{RHIVertexSemantics::Texcoord, 3, 0, RHIFormat::R32G32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, texcoord0)},
			    VertexInputState::InputAttribute{RHIVertexSemantics::Texcoord, 4, 0, RHIFormat::R32G32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, texcoord1)},
			};
			mesh_visibility_pipeline.pipeline->SetVertexInputState(vertex_input_state);

			mesh_visibility_pipeline.meta = renderer->RequireShaderMeta(vertex_shader);
			mesh_visibility_pipeline.meta += renderer->RequireShaderMeta(frag_shader);
		}

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto  visibility_buffer = render_graph.GetTexture(desc.resources.at("Visibility Buffer").handle);
			auto  depth_stencil     = render_graph.GetTexture(desc.resources.at("Depth Stencil").handle);
			auto *rhi_context       = renderer->GetRHIContext();
			auto *gpu_scene         = black_board.Get<GPUScene>();
			auto *view              = black_board.Get<View>();

			mesh_visibility_pipeline.render_target->Clear();
			mesh_visibility_pipeline.render_target->Set(0, visibility_buffer, TextureRange{}, ColorAttachment{});
			mesh_visibility_pipeline.render_target->Set(depth_stencil, TextureRange{}, DepthStencilAttachment{});

			// Mesh Shading
			if (rhi_context->IsFeatureSupport(RHIFeature::MeshShading))
			{
				// Clear
				if (gpu_scene->mesh_buffer.instance_count == 0 &&
				    gpu_scene->skinned_mesh_buffer.instance_count == 0)
				{
					cmd_buffer->SetViewport(static_cast<float>(mesh_visibility_pipeline.render_target->GetWidth()), static_cast<float>(mesh_visibility_pipeline.render_target->GetHeight()));
					cmd_buffer->SetScissor(mesh_visibility_pipeline.render_target->GetWidth(), mesh_visibility_pipeline.render_target->GetHeight());
					cmd_buffer->BeginRenderPass(mesh_visibility_pipeline.render_target.get());
					cmd_buffer->EndRenderPass();
				}

				// Draw Mesh
				if (gpu_scene->mesh_buffer.instance_count > 0)
				{
					auto *descriptor = rhi_context->CreateDescriptor(mesh_visibility_pipeline.meta);
					descriptor->BindBuffer("InstanceBuffer", gpu_scene->mesh_buffer.instances.get())
					    .BindBuffer("ViewBuffer", view->buffer.get())
					    .BindBuffer("VertexBuffer", gpu_scene->mesh_buffer.vertex_buffers)
					    .BindBuffer("IndexBuffer", gpu_scene->mesh_buffer.index_buffers)
					    .BindBuffer("MeshletBuffer", gpu_scene->mesh_buffer.meshlet_buffers)
					    .BindBuffer("MeshletDataBuffer", gpu_scene->mesh_buffer.meshlet_data_buffers);

					cmd_buffer->SetViewport(static_cast<float>(mesh_visibility_pipeline.render_target->GetWidth()), static_cast<float>(mesh_visibility_pipeline.render_target->GetHeight()));
					cmd_buffer->SetScissor(mesh_visibility_pipeline.render_target->GetWidth(), mesh_visibility_pipeline.render_target->GetHeight());
					cmd_buffer->BeginRenderPass(mesh_visibility_pipeline.render_target.get());
					cmd_buffer->BindDescriptor(descriptor);
					cmd_buffer->BindPipelineState(mesh_visibility_pipeline.pipeline.get());
					cmd_buffer->DrawMeshTask(gpu_scene->mesh_buffer.max_meshlet_count, gpu_scene->mesh_buffer.instance_count, 1, 32, 1, 1);
					cmd_buffer->EndRenderPass();
				}

				// Draw Skinned Mesh
			}
			else
			{
				// Draw Mesh
				if (gpu_scene->mesh_buffer.instance_count > 0)
				{
					auto meshes = renderer->GetScene()->GetComponents<Cmpt::MeshRenderer>();

					auto *descriptor = rhi_context->CreateDescriptor(mesh_visibility_pipeline.meta);
					descriptor->BindBuffer("InstanceBuffer", gpu_scene->mesh_buffer.instances.get())
					    .BindBuffer("ViewBuffer", view->buffer.get())
					    .BindBuffer("VertexBuffer", gpu_scene->mesh_buffer.vertex_buffers)
					    .BindBuffer("IndexBuffer", gpu_scene->mesh_buffer.index_buffers);

					cmd_buffer->SetViewport(static_cast<float>(mesh_visibility_pipeline.render_target->GetWidth()), static_cast<float>(mesh_visibility_pipeline.render_target->GetHeight()));
					cmd_buffer->SetScissor(mesh_visibility_pipeline.render_target->GetWidth(), mesh_visibility_pipeline.render_target->GetHeight());
					cmd_buffer->BeginRenderPass(mesh_visibility_pipeline.render_target.get());
					cmd_buffer->BindDescriptor(descriptor);
					cmd_buffer->BindPipelineState(mesh_visibility_pipeline.pipeline.get());

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
								cmd_buffer->DrawIndexed(resource->GetIndexCount(), 1, 0, 0, instance_id);
							}
						}
					}

					cmd_buffer->EndRenderPass();
				}

				// Draw Skinned Mesh
			}
		};
	}

	virtual void OnImGui(Variant *config)
	{
	}
};

CONFIGURATION_PASS(VisibilityGeometryPass)