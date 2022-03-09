#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class WireFramePass : public TRenderPass<WireFramePass>
{
  public:
	WireFramePass() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	bool  m_enable     = false;
	float m_line_width = 2.f;
};
}        // namespace Ilum::pass