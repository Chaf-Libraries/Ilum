#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class Path : public TRenderPass<Path>
{
  public:
	Path() = default;

	~Path() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	bool m_finish = false;

	struct
	{
		uint32_t anti_alias              = 0;
		uint32_t directional_light_count = 0;
		uint32_t spot_light_count        = 0;
		uint32_t point_light_count       = 0;
		uint32_t area_light_count       = 0;
		int32_t  max_bounce              = 5;
		float    firefly_clamp_threshold = 4.f;
		float    parameter               = 0.1f;
	} m_push_block;

	int32_t m_max_spp = 100;
	bool    m_update  = true;
};
}        // namespace Ilum::pass