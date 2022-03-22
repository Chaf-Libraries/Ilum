#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class CubemapPrefilter : public TRenderPass<CubemapPrefilter>
{
  public:
	CubemapPrefilter();

	~CubemapPrefilter();

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	bool m_update = false;

	std::vector<VkDescriptorSet> m_descriptor_sets;
	std::vector<VkImageView>     m_views;
	const uint32_t               m_mip_levels    = 5;
	int32_t                     m_current_level = 0;

	struct
	{
		VkExtent2D cubemap_extent = {1024, 1024};
		VkExtent2D mip_extent = {};
		float      roughness      = 0.f;
	} m_push_data;
};
}        // namespace Ilum::pass