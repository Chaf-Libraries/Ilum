#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
// Extract bright part for blooming
class BrightPass : public TRenderPass<BrightPass>
{
  public:
	BrightPass(const std::string &input);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::string m_input;

	float m_threshold = 0.75f;
	uint32_t m_enable = 0;
};
}        // namespace Ilum::pass