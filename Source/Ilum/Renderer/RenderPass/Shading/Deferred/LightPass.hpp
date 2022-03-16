#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

#include <glm/glm.hpp>

namespace Ilum::pass
{
class LightPass : public TRenderPass<LightPass>
{
  public:
	LightPass();

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	struct
	{
		uint32_t   directional_light_count = 0;
		uint32_t   spot_light_count        = 0;
		uint32_t   point_light_count       = 0;
		uint32_t   enable_multi_bounce     = 0;
		VkExtent2D extent                  = {};
	} m_push_block;

	Sampler m_shadowmap_sampler;
};
}        // namespace Ilum::pass