#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class SurfaceViewPass : public TRenderPass<SurfaceViewPass>
{
  public:
	SurfaceViewPass() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;
};
}        // namespace Ilum::pass