#pragma once

#include "Renderer/RenderPass.hpp"

namespace Ilum
{
STRUCT(ReblurBlur, Enable, RenderPass("Reblur Blur"), Category("Nvidia Ray Tracing Denoisor")) :
    public RenderPass
{
	virtual RenderPassDesc CreateDesc() override;

	virtual RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) override;
};
}        // namespace Ilum