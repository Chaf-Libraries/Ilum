#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
// Extract bright part for blooming
class BloomMask : public TRenderPass<BloomMask>
{
  public:
	BloomMask(const std::string &input, const std::string &output);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::string m_input;
	std::string m_output;

	float m_threshold = 0.75f;
	uint32_t m_enable = 1;
};
}        // namespace Ilum::pass