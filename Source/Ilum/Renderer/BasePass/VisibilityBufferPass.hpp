#pragma once

#include <RenderCore/RenderGraph/RenderGraph.hpp>
#include <RenderCore/RenderGraph/RenderGraphBuilder.hpp>

#include "Renderer/Renderer.hpp"

namespace Ilum::Pass
{
class VisibilityBufferPass
{
  public:
	static RenderPassDesc CreateDesc();

	static RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer);

	struct Config
	{
		float a = 100.f;
		std::string m = "fuck you ";
	};
};

REFLECTION_BEGIN(VisibilityBufferPass::Config)
REFLECTION_PROPERTY(a)
REFLECTION_PROPERTY(m)
REFLECTION_END()

RENDER_PASS_REGISTERATION(VisibilityBufferPass);
}        // namespace Ilum::Pass