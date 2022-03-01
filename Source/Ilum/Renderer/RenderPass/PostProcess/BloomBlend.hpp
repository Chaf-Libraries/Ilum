#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
// Extract bright part for blooming
class BloomBlend : public TRenderPass<BloomBlend>
{
  public:
	BloomBlend(const std::string &input, const std::string &output);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

  private:
	std::string m_input;
	std::string m_output;
	std::string m_result;
};
}        // namespace Ilum::pass