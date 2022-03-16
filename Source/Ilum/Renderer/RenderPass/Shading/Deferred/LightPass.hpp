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
		int32_t   filter_method              = 1;		// 0 - no filter, 1 - PCF, 2 - PCSS
		float      sample_scale        = 3.f;
		int32_t    sample_num          = 20;
		int32_t    sample_method       = 1;// 0 - uniform, 1 - poisson
		float      light_size                 = 10.f;
	} m_push_block;

	Sampler m_shadowmap_sampler;
};
}        // namespace Ilum::pass