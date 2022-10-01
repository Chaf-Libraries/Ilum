#pragma once

#include "Renderer/RenderPass.hpp"

namespace Ilum
{
struct [[RenderPass("Visibility Buffer Visualization")]] VisibilityBufferVisualization : public RenderPass
{
	virtual RenderPassDesc CreateDesc() override;

	virtual RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) override;
};
}        // namespace Ilum