#pragma once

#include <RenderCore/RenderGraph/RenderGraph.hpp>
#include <RenderCore/RenderGraph/RenderGraphBuilder.hpp>

#include "Renderer/Renderer.hpp"

namespace Ilum::Pass
{
class VisibilityBufferPass
{
  public:
	static RenderPassDesc CreateDesc(size_t &handle);

	static RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer);

	struct Config
	{
	};
};

RENDER_PASS_REGISTERATION(VisibilityBufferPass);
}        // namespace Ilum::Pass