#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
// Extract bright part for blooming
class BlendPass : public TRenderPass<BlendPass>
{
  public:
	BlendPass(const std::string &src1, const std::string &src2, const std::string &result);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

  private:
	std::string m_src1;
	std::string m_src2;
	std::string m_result;
};
}        // namespace Ilum::pass