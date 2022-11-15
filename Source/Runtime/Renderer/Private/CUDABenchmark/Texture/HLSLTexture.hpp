#pragma once

#include "Renderer/RenderPass.hpp"

namespace Ilum
{
STRUCT(HLSLTexture, Enable, RenderPass("HLSL Texture"), Category("CUDA-HLSL Test")) :
    public RenderPass
{
	virtual RenderPassDesc CreateDesc() override;

	virtual RenderGraph::RenderTask Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) override;
};
}        // namespace Ilum