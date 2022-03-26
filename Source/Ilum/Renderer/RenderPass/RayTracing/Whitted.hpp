#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class Whitted : public TRenderPass<Whitted>
{
  public:
	Whitted() = default;

	~Whitted() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	bool m_finish = false;

	struct
	{
		uint32_t directional_light_count = 0;
		uint32_t spot_light_count        = 0;
		uint32_t point_light_count       = 0;
	} m_push_block;
};
}        // namespace Ilum::pass