#pragma once

#include <RenderCore/RenderGraph/RenderGraph.hpp>
#include <RenderCore/RenderGraph/RenderGraphBuilder.hpp>

#include "Renderer/Renderer.hpp"

namespace Ilum
{
class VisibilityBufferPass
{
  public:
	static RenderPassDesc CreateDesc();

	static RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer);
};

REFLECTION_STRUCT VisibilityBufferPass_Config
{
	REFLECTION_PROPERTY()
	float a;

	REFLECTION_PROPERTY()
	std::string m;

	VisibilityBufferPass_Config() :
	    a(100.f),
	    m("fuck you")
	{
	}
};

RENDER_PASS_REGISTERATION(VisibilityBufferPass);
}        // namespace Ilum