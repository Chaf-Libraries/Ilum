#include "TrianglePass.hpp"

#include <Scene/Component/TransformComponent.hpp>
#include <Scene/Entity.hpp>
#include <Scene/Scene.hpp>

namespace Ilum
{
RenderPassDesc TrianglePass::CreateDesc()
{
	RenderPassDesc desc = {};

	desc
	    .SetName<TrianglePass>()
	    .SetBindPoint(BindPoint::Rasterization)
	    .Write("Output", RenderResourceDesc::Type::Texture, RHIResourceState::RenderTarget)
	    .SetConfig(Config());

	return desc;
}

RenderGraph::RenderTask TrianglePass::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	auto *vertex_shader   = renderer->RequireShader("Source/Shaders/DrawUV.hlsl", "VSmain", RHIShaderStage::Vertex);
	auto *fragment_shader = renderer->RequireShader("Source/Shaders/DrawUV.hlsl", "PSmain", RHIShaderStage::Fragment);

	ShaderMeta meta;
	meta += renderer->RequireShaderMeta(vertex_shader);
	meta += renderer->RequireShaderMeta(fragment_shader);

	std::shared_ptr<RHIDescriptor>    descriptor     = std::move(renderer->GetRHIContext()->CreateDescriptor(meta));
	std::shared_ptr<RHIPipelineState> pipeline_state = std::move(renderer->GetRHIContext()->CreatePipelineState());
	std::shared_ptr<RHIRenderTarget>  render_target  = std::move(renderer->GetRHIContext()->CreateRenderTarget());

	pipeline_state->SetShader(RHIShaderStage::Vertex, vertex_shader);
	pipeline_state->SetShader(RHIShaderStage::Fragment, fragment_shader);

	BlendState bend_state;
	bend_state.attachment_states.resize(1);
	pipeline_state->SetBlendState(bend_state);

	RasterizationState rasterization_state;
	rasterization_state.cull_mode  = RHICullMode::None;
	rasterization_state.front_face = RHIFrontFace::Clockwise;
	pipeline_state->SetRasterizationState(rasterization_state);

	float factor = 1.f;

	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		Config config_ = config.convert<Config>();

		render_target->Clear()
		    .Set(0, render_graph.GetTexture(desc.resources.at("Output").handle), RHITextureDimension::Texture2D, ColorAttachment{});
		cmd_buffer->BindDescriptor(descriptor.get());
		cmd_buffer->SetViewport(static_cast<float>(render_target->GetWidth()), static_cast<float>(render_target->GetHeight()));
		cmd_buffer->SetScissor(render_target->GetWidth(), render_target->GetHeight());
		cmd_buffer->BeginRenderPass(render_target.get());
		descriptor->SetConstant("a", config_.a).BindBuffer("View", renderer->GetViewBuffer());
		renderer->GetScene()->Execute([&](Entity &entity) {
			auto transform = entity.GetComponent<TransformComponent>().world_transform;
			descriptor->SetConstant("transform", transform);
			cmd_buffer->BindPipelineState(pipeline_state.get());
			cmd_buffer->Draw(3, 1);
		});
		cmd_buffer->EndRenderPass();
	};
}
}        // namespace Ilum