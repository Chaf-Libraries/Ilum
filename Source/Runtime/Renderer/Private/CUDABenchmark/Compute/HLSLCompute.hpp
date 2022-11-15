#pragma once

#include "Renderer/RenderPass.hpp"

namespace Ilum
{
STRUCT(HLSLCompute, Enable, RenderPass("HLSL Compute"), Category("CUDA-HLSL Test")) :
    public RenderPass
{
	virtual RenderPassDesc CreateDesc() override;

	virtual RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) override;
};
}        // namespace Ilum