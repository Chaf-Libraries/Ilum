#pragma once

#include "Renderer/RenderPass.hpp"

namespace Ilum
{
STRUCT(PathTracing, Enable, RenderPass("Path Tracing"), Category("Ray Tracing")) :
    public RenderPass
{
	virtual RenderPassDesc CreateDesc() override;

	virtual RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) override;
};
}        // namespace Ilum