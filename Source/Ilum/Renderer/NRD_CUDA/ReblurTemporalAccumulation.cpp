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
	auto *shader = renderer->RequireShader("Source/Shaders/NRD/ReblurTemporalAccumulation.hlsl", "MainCS", RHIShaderStage::Compute, {}, true);

	ShaderMeta meta = renderer->RequireShaderMeta(shader);

	std::shared_ptr<std::map<RHITexture *, std::unique_ptr<RHITexture>>> texture_map = std::make_shared<std::map<RHITexture *, std::unique_ptr<RHITexture>>>();

	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		Config config_ = config.convert<Config>();

		auto *normal_roughness = render_graph.GetTexture(desc.resources.at("In_Normal_Roughness").handle);
		if ((*texture_map).find(normal_roughness) == (*texture_map).end())
		{
			(*texture_map).emplace(normal_roughness, renderer->GetRHIContext()->MapToCUDATexture(normal_roughness));
		}
	};
}
}        // namespace Ilum