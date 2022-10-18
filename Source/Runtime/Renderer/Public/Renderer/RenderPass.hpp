#pragma once

#include <RenderCore/RenderGraph/RenderGraph.hpp>
#include <RenderCore/RenderGraph/RenderGraphBuilder.hpp>

#include "Renderer.hpp"

namespace Ilum
{
struct RenderPass
{
	virtual RenderPassDesc CreateDesc() = 0;

	virtual RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) = 0;
};
}        // namespace Ilum