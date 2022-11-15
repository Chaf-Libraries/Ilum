#pragma once

#include "Renderer/RenderPass.hpp"

namespace Ilum
{
STRUCT(CUDACompute, Enable, RenderPass("CUDA Compute"), Category("CUDA-HLSL Test")) :
    public RenderPass
{
	virtual RenderPassDesc CreateDesc() override;

	virtual RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) override;
};
}        // namespace Ilum