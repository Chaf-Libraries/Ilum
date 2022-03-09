#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class HizPass : public TRenderPass<HizPass>
{
  public:
	HizPass();

	~HizPass();

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::vector<DescriptorSet> m_descriptor_sets;
	std::vector<VkImageView>   m_views;
	VkSampler                  m_hiz_sampler=VK_NULL_HANDLE;
	Image                      m_hiz;
	Image                      m_linear_depth;
	int32_t                    m_current_level = 0;
};
}        // namespace Ilum::pass