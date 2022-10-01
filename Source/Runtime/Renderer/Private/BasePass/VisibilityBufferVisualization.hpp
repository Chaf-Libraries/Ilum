#pragma once

#include "Renderer/RenderPass.hpp"

namespace Ilum
{
STRUCT(VisibilityBufferVisualization, Enable, RenderPass("Visibility Buffer Visualization"), Category("Visualization")) :
    public RenderPass
{
	virtual RenderPassDesc CreateDesc() override;

	virtual RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) override;
};
}        // namespace Ilum