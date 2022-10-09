#pragma once

#include "Renderer/RenderPass.hpp"

namespace Ilum
{
STRUCT(VisibilityBufferPass, Enable, RenderPass("Visibility Buffer Pass"), Category("Base Pass")) :
    public RenderPass
{
	virtual RenderPassDesc CreateDesc() override;

	virtual RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) override;
};
}        // namespace Ilum