#pragma once

#include "ReblurTemporalAccumulation.hpp"

namespace Ilum
{
RenderPassDesc ReblurTemporalAccumulation::CreateDesc()
{
	RenderPassDesc desc = {};
	desc.SetName("ReblurTemporalAccumulation")
	    .SetBindPoint(BindPoint::CUDA);

	return desc;
}

RenderGraph::RenderTask ReblurTemporalAccumulation::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	auto *shader = renderer->RequireShader("Source/Shaders/NRD/ReblurTemporalAccumulation.hlsl", "MainCS", RHIShaderStage::Compute, {}, true);

	ShaderMeta meta = renderer->RequireShaderMeta(shader);

	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		Config config_ = config.convert<Config>();

		
	};
}
}        // namespace Ilum