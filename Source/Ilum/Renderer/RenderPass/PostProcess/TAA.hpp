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
	TAAPass(const std::string &input, const std::string &prev, const std::string &output);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::string m_input;
	std::string m_prev;
	std::string m_output;

	struct
	{
		float    feedback_min = 0.925f;
		float    feedback_max = 0.987f;
		uint32_t sharpen      = 0;
	} m_push_constants;
};
}        // namespace Ilum::pass