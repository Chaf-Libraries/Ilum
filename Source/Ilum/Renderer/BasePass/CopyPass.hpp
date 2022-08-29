#pragma once

#include <RenderCore/RenderGraph/RenderGraph.hpp>
#include <RenderCore/RenderGraph/RenderGraphBuilder.hpp>

#include "Renderer/Renderer.hpp"

namespace Ilum
{
class CopyPass
{
  public:
	static RenderPassDesc CreateDesc();

	static RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		return [=](RenderGraph &, RHICommand *) {};
	}
};

RENDER_PASS_REGISTERATION(CopyPass);
}        // namespace Ilum::Pass