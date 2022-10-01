#pragma once

#include "Renderer/RenderPass.hpp"

namespace Ilum
{
STRUCT(TrianglePass, Enable, RenderPass("Triangle Pass"), Category("Base Pass")) :
    public RenderPass
{
	virtual RenderPassDesc CreateDesc() override;

	virtual RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) override;

	STRUCT(Config, Enable)
	{
		float a = 1.f;
	};
};
}        // namespace Ilum