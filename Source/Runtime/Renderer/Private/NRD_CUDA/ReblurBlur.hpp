#pragma once

#include "Renderer/RenderPass.hpp"

namespace Ilum
{
struct [[RenderPass("Reblur Blur"), Category("Nvidia Ray Tracing Denoisor")]] ReblurBlur : public RenderPass
{
	virtual RenderPassDesc CreateDesc() override;

	virtual RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) override;
};
}        // namespace Ilum