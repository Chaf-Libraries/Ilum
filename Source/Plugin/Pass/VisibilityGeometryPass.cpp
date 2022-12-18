#include "IPass.hpp"

#include <Core/Core.hpp>
#include <RenderGraph/RenderGraph.hpp>
#include <RenderGraph/RenderGraphBuilder.hpp>
#include <Renderer/RenderData.hpp>
#include <Renderer/Renderer.hpp>

#include <imgui.h>

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

		{
			mesh_visibility_pipeline.pipeline      = std::shared_ptr<RHIPipelineState>(std::move(renderer->GetRHIContext()->CreatePipelineState()));
			mesh_visibility_pipeline.render_target = std::shared_ptr<RHIRenderTarget>(std::move(renderer->GetRHIContext()->CreateRenderTarget()));

			auto *task_shader = renderer->RequireShader("Source/Shaders/VisibilityMeshPass.hlsl", "ASmain", RHIShaderStage::Task);
			auto *mesh_shader = renderer->RequireShader("Source/Shaders/VisibilityMeshPass.hlsl", "MSmain", RHIShaderStage::Mesh);
			auto *frag_shader = renderer->RequireShader("Source/Shaders/VisibilityMeshPass.hlsl", "PSmain", RHIShaderStage::Fragment);

			mesh_visibility_pipeline.pipeline->SetShader(RHIShaderStage::Task, task_shader);
			mesh_visibility_pipeline.pipeline->SetShader(RHIShaderStage::Mesh, mesh_shader);
			mesh_visibility_pipeline.pipeline->SetShader(RHIShaderStage::Fragment, frag_shader);

			BlendState blend_state;
			blend_state.attachment_states.resize(1);
			mesh_visibility_pipeline.pipeline->SetBlendState(blend_state);

			mesh_visibility_pipeline.meta = renderer->RequireShaderMeta(task_shader);
			mesh_visibility_pipeline.meta += renderer->RequireShaderMeta(mesh_shader);
			mesh_visibility_pipeline.meta += renderer->RequireShaderMeta(frag_shader);
		}

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto  visibility_buffer = render_graph.GetTexture(desc.resources.at("Visibility Buffer").handle);
			auto  depth_stencil     = render_graph.GetTexture(desc.resources.at("Depth Stencil").handle);
			auto *rhi_context       = renderer->GetRHIContext();
			auto *gpu_scene         = black_board.Get<GPUScene>();
			auto *view              = black_board.Get<View>();

			// Mesh Shading
			if (rhi_context->IsFeatureSupport(RHIFeature::MeshShading))
			{
				// Draw Mesh
				if (gpu_scene->mesh_buffer.instance_count > 0)
				{
					mesh_visibility_pipeline.render_target->Clear();
					mesh_visibility_pipeline.render_target->Set(0, visibility_buffer, TextureRange{}, ColorAttachment{});
					mesh_visibility_pipeline.render_target->Set(depth_stencil, TextureRange{}, DepthStencilAttachment{});

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
					cmd_buffer->DrawMeshTask(gpu_scene->mesh_buffer.instance_count, gpu_scene->mesh_buffer.max_meshlet_count, 1, 8, 8, 1);
					cmd_buffer->EndRenderPass();
				}

				// Draw Skinned Mesh
			}
			else
			{
				// Draw Mesh

				// Draw Skinned Mesh
			}
		};
	}

	virtual void OnImGui(Variant *config)
	{
	}
};

CONFIGURATION_PASS(VisibilityGeometryPass)