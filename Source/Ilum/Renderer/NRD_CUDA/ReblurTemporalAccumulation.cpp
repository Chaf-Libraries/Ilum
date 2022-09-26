#pragma once

#include "ReblurTemporalAccumulation.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
RenderPassDesc ReblurTemporalAccumulation::CreateDesc()
{
	RenderPassDesc desc = {};
	desc.SetName("ReblurTemporalAccumulation")
	    .Read("In_Normal_Roughness", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
	    .Read("In_ViewZ", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
	    .Read("In_ObjectMotion", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
	    .Read("In_Prev_ViewZ", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
	    .Read("In_Prev_Normal_Roughness", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
	    .Read("In_Prev_AccumSpeeds_MaterialID", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
	    .Read("In_Diff", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
	    .Read("In_Diff_History", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
	    .Write("Out_Diff", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess)
	    .Write("Out_Data1", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess)
	    .SetBindPoint(BindPoint::CUDA);

	return desc;
}

RenderGraph::RenderTask ReblurTemporalAccumulation::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	// auto *shader = renderer->RequireShader("Source/Shaders/NRD/ReblurTemporalAccumulation.hlsl", "MainCS", RHIShaderStage::Compute, {}, true);
	auto *shader = renderer->RequireShader("Source/Shaders/NRD/Test.hlsl", "MainCS", RHIShaderStage::Compute, {}, true);

	ShaderMeta meta = renderer->RequireShaderMeta(shader);

	std::shared_ptr<RHIDescriptor>    descriptor     = std::move(renderer->GetRHIContext()->CreateDescriptor(meta, true));
	std::shared_ptr<RHIPipelineState> pipeline_state = std::move(renderer->GetRHIContext()->CreatePipelineState(true));

	pipeline_state->SetShader(RHIShaderStage::Compute, shader);

	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		Config config_ = config.convert<Config>();

		auto *normal_roughness = render_graph.GetCUDATexture(desc.resources.at("Out_Diff").handle);
		descriptor->BindTexture("OutputImage", normal_roughness, RHITextureDimension::Texture2D);

		cmd_buffer->BindDescriptor(descriptor.get());
		cmd_buffer->BindPipelineState(pipeline_state.get());
		cmd_buffer->Dispatch(normal_roughness->GetDesc().width, normal_roughness->GetDesc().height, 1, 8, 8, 1);
	};
}
}        // namespace Ilum