#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class HizPass : public TRenderPass<HizPass>
{
  public:
	HizPass() = default;

	~HizPass();

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	int32_t m_current_level = 0;

	std::vector<VkImageView> m_views;
};
}        // namespace Ilum::pass