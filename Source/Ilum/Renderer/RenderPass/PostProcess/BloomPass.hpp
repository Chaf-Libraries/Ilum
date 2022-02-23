#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
// Extract bright part for blooming
class BloomPass : public TRenderPass<BloomPass>
{
  public:
	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	struct
	{
		float    threshold = 0.75f;
		float    scale     = 3.f;
		float    strength  = 0.13f;
		uint32_t enable    = 0;
	} m_bloom_data;
};
}        // namespace Ilum::pass