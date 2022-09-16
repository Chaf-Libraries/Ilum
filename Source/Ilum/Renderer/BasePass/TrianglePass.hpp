#pragma once

#include "Renderer/RenderPass.hpp"

namespace Ilum
{
struct [[RenderPass("Triangle Pass"), Category("Basic Pass")]] TrianglePass : public RenderPass
{
	virtual RenderPassDesc CreateDesc() override;

	virtual RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) override;

	struct Config
	{
		float a = 1.f;
	};
};
}        // namespace Ilum