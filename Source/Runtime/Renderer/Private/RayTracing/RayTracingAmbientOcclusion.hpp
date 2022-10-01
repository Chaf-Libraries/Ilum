#pragma once

#include "Renderer/RenderPass.hpp"

namespace Ilum
{
struct [[RenderPass("Ray Tracing Ambient Occlusion"), Category("Ray Tracing")]] RayTracingAmbientOcclusion : public RenderPass
{
	virtual RenderPassDesc CreateDesc() override;

	virtual RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) override;
};
}        // namespace Ilum