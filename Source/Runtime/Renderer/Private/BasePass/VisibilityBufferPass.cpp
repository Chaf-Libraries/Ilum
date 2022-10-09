#include "VisibilityBufferPass.hpp"

#include <Resource/ResourceManager.hpp>
#include <Scene/Component/AllComponent.hpp>
#include <Scene/Scene.hpp>

namespace Ilum
{
RenderPassDesc VisibilityBufferPass::CreateDesc()
{
	RenderPassDesc desc = {};

	desc.name = "VisibilityBufferPass";
	desc
	    .SetName("VisibilityBufferPass")
	    .SetBindPoint(BindPoint::Rasterization)
	    .Write("VisibilityBuffer", RenderResourceDesc::Type::Texture, RHIResourceState::RenderTarget)
	    .Write("DepthBuffer", RenderResourceDesc::Type::Texture, RHIResourceState::DepthWrite);

	return desc;
}

RenderGraph::RenderTask VisibilityBufferPass::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	ShaderMeta meta;

	std::shared_ptr<RHIPipelineState> pipeline_state = std::move(renderer->GetRHIContext()->CreatePipelineState());
	std::shared_ptr<RHIRenderTarget>  render_target  = std::move(renderer->GetRHIContext()->CreateRenderTarget());

	// Use mesh shader
	if (renderer->GetRHIContext()->IsFeatureSupport(RHIFeature::MeshShading))
	{
		auto *task_shader = renderer->RequireShader("Source/Shaders/VisibilityBuffer.hlsl", "ASmain", RHIShaderStage::Task);
		auto *mesh_shader = renderer->RequireShader("Source/Shaders/VisibilityBuffer.hlsl", "MSmain", RHIShaderStage::Mesh);
		auto *frag_shader = renderer->RequireShader("Source/Shaders/VisibilityBuffer.hlsl", "PSmain", RHIShaderStage::Fragment);

		meta += renderer->RequireShaderMeta(task_shader);
		meta += renderer->RequireShaderMeta(mesh_shader);
		meta += renderer->RequireShaderMeta(frag_shader);

		pipeline_state->SetShader(RHIShaderStage::Task, task_shader);
		pipeline_state->SetShader(RHIShaderStage::Mesh, mesh_shader);
		pipeline_state->SetShader(RHIShaderStage::Fragment, frag_shader);
	}

	std::shared_ptr<RHIDescriptor> descriptor = std::move(renderer->GetRHIContext()->CreateDescriptor(meta));

	BlendState bend_state = {};
	bend_state.attachment_states.resize(1);
	pipeline_state->SetBlendState(bend_state);

	RasterizationState rasterization_state;
	rasterization_state.cull_mode  = RHICullMode::None;
	rasterization_state.front_face = RHIFrontFace::Clockwise;
	pipeline_state->SetRasterizationState(rasterization_state);

	DepthStencilState depth_stencil_state  = {};
	depth_stencil_state.depth_write_enable = true;
	depth_stencil_state.depth_test_enable  = true;
	pipeline_state->SetDepthStencilState(depth_stencil_state);

	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		auto *visibility_buffer = render_graph.GetTexture(desc.resources.at("VisibilityBuffer").handle);
		auto *depth_buffer      = render_graph.GetTexture(desc.resources.at("DepthBuffer").handle);

		render_target->Clear()
		    .Set(0, visibility_buffer, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, ColorAttachment{})
		    .Set(depth_buffer, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, DepthStencilAttachment{});

		cmd_buffer->SetViewport(static_cast<float>(render_target->GetWidth()), static_cast<float>(render_target->GetHeight()));
		cmd_buffer->SetScissor(render_target->GetWidth(), render_target->GetHeight());
		cmd_buffer->BeginRenderPass(render_target.get());

		const auto &scene_info = renderer->GetSceneInfo();

		descriptor
		    ->BindBuffer("View", renderer->GetViewBuffer())
		    .BindBuffer("VertexBuffer", scene_info.static_vertex_buffers)
		    .BindBuffer("IndexBuffer", scene_info.static_index_buffers)
		    .BindBuffer("MeshletVertexBuffer", scene_info.meshlet_vertex_buffers)
		    .BindBuffer("MeshletPrimitiveBuffer", scene_info.meshlet_primitive_buffers)
		    .BindBuffer("MeshletBuffer", scene_info.meshlet_buffers)
		    .BindBuffer("InstanceBuffer", scene_info.instance_buffer.get());

		for (uint32_t i = 0; i < scene_info.meshlet_count.size(); i++)
		{
			descriptor->SetConstant("instance_idx", i);
			cmd_buffer->BindDescriptor(descriptor.get());
			cmd_buffer->BindPipelineState(pipeline_state.get());
			cmd_buffer->DrawMeshTask(scene_info.meshlet_count[i], 32);
		}
		cmd_buffer->EndRenderPass();
	};
}

}        // namespace Ilum