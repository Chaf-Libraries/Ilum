#pragma once

#include "Renderer/RenderPass.hpp"

namespace Ilum
{
STRUCT(PresentPass, Enable, RenderPass("Present Pass"), Category("Basic Pass")) :
    public RenderPass
{
	virtual RenderPassDesc CreateDesc() override;

	virtual RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) override;
};
}        // namespace Ilum