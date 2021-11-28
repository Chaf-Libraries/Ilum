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

  private:
	std::vector<DescriptorSet> m_descriptor_sets;
	std::vector<VkImageView>   m_views;
};
}        // namespace Ilum::pass