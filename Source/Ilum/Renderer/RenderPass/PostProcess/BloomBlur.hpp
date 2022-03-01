#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
// Extract bright part for blooming
class BloomBlur : public TRenderPass<BloomBlur>
{
  public:
	BloomBlur(const std::string &input, const std::string &output, bool horizontal = false);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::string m_input;
	std::string m_output;

	struct
	{
		VkExtent2D extent;
		float      scale      = 2.f;
		float      strength   = 1.f;
		uint32_t   horizontal = 0;
	} m_push_data;

	bool m_enable = true;
};
}        // namespace Ilum::pass