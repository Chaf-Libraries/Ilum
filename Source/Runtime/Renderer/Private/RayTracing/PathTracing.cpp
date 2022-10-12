#include "PathTracing.hpp"

namespace Ilum
{
RenderPassDesc PathTracing::CreateDesc()
{
	RenderPassDesc desc;
	desc.SetName("PathTracing")
	    .SetBindPoint(BindPoint::RayTracing)
	    .Write("Output", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess);

	return desc;
}

RenderGraph::RenderTask PathTracing::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	ShaderMeta meta;

	std::shared_ptr<RHIPipelineState> pipeline_state = std::move(renderer->GetRHIContext()->CreatePipelineState());

	auto *ray_gen_shader     = renderer->RequireShader("Source/Shaders/PathTracing.hlsl", "RayGenMain", RHIShaderStage::RayGen);
	auto *closest_hit_shader = renderer->RequireShader("Source/Shaders/PathTracing.hlsl", "ClosesthitMain", RHIShaderStage::ClosestHit);
	auto *miss_shader        = renderer->RequireShader("Source/Shaders/PathTracing.hlsl", "MissMain", RHIShaderStage::Miss);

	pipeline_state->SetShader(RHIShaderStage::RayGen, ray_gen_shader);
	pipeline_state->SetShader(RHIShaderStage::ClosestHit, closest_hit_shader);
	pipeline_state->SetShader(RHIShaderStage::Miss, miss_shader);

	meta += renderer->RequireShaderMeta(ray_gen_shader);
	meta += renderer->RequireShaderMeta(closest_hit_shader);
	meta += renderer->RequireShaderMeta(miss_shader);

	std::shared_ptr<RHIDescriptor> descriptor = std::move(renderer->GetRHIContext()->CreateDescriptor(meta));

	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		auto *output = render_graph.GetTexture(desc.resources.at("Output").handle);

		const auto &scene_info = renderer->GetSceneInfo();
		descriptor
		    ->BindBuffer("View", renderer->GetViewBuffer())
		    .BindTexture("OutputImage", output, RHITextureDimension::Texture2D)
		    .BindAccelerationStructure("TopLevelAS", scene_info.top_level_as)
		    .BindBuffer("VertexBuffer", scene_info.static_vertex_buffers)
		    .BindBuffer("IndexBuffer", scene_info.static_index_buffers)
		    .BindBuffer("MeshletVertexBuffer", scene_info.meshlet_vertex_buffers)
		    .BindBuffer("MeshletPrimitiveBuffer", scene_info.meshlet_primitive_buffers)
		    .BindBuffer("MeshletBuffer", scene_info.meshlet_buffers)
		    .BindBuffer("InstanceBuffer", scene_info.instance_buffer.get());

		cmd_buffer->BindDescriptor(descriptor.get());
		cmd_buffer->BindPipelineState(pipeline_state.get());
		cmd_buffer->TraceRay(output->GetDesc().width, output->GetDesc().height, 1);
	};
}
}        // namespace Ilum