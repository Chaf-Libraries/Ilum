#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

#include <glm/glm.hpp>

namespace Ilum::pass
{
// Extract bright part for blooming
class TAAPass : public TRenderPass<TAAPass>
{
  public:
	TAAPass();

	virtual void onUpdate() override;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	bool m_enable = true;

	glm::vec2 m_current_jitter = glm::vec2(0.f);
	glm::vec2 m_prev_jitter    = glm::vec2(0.f);
	glm::vec2 m_feedback       = glm::vec2(1.f, 1.f);

	std::vector<glm::vec2> m_jitter_samples;
};
}        // namespace Ilum::pass