#pragma once

#include <RenderCore/RenderGraph/RenderGraph.hpp>
#include <RenderCore/RenderGraph/RenderGraphBuilder.hpp>

#include "Renderer/Renderer.hpp"

namespace Ilum::Pass
{
class CopyPass
{
  public:
	static RenderPassDesc CreateDesc(size_t &handle);

	static void Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
	}
};

RENDER_PASS_REGISTERATION(CopyPass);
}        // namespace Ilum::Pass