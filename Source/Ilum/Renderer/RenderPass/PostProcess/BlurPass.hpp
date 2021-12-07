#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
// Extract bright part for blooming
class BlurPass : public TRenderPass<BlurPass>
{
  public:
	BlurPass(const std::string &input, const std::string &output, bool horizental = false);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

  private:
	std::string m_input;
	std::string m_output;
	bool m_horizental;
};
}        // namespace Ilum::pass