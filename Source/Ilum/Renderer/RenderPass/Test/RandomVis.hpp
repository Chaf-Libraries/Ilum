#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class RandomVis : public TRenderPass<RandomVis>
{
  public:
	RandomVis() = default;

	~RandomVis() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	struct
	{
		VkExtent2D extent = {};
		uint32_t frame = 0;
		int32_t    m_sample_count  = 3000;
		int32_t   sampling_method = 0;
	}m_push_data;

	bool m_update = false;
};
}        // namespace Ilum::pass