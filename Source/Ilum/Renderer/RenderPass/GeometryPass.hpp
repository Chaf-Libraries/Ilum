#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class GeometryPass : public TRenderPass<GeometryPass>
{
  public:
	GeometryPass(const std::string &output);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

  private:
	std::string m_output;
};

// TODO: Bindless Geometry Pass
}        // namespace Ilum::pass