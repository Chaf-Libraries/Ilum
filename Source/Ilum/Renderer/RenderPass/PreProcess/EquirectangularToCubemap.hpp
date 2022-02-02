#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class EquirectangularToCubemap : public TRenderPass<EquirectangularToCubemap>
{
  public:
	EquirectangularToCubemap() = default;

	~EquirectangularToCubemap();

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

  private:
	std::vector<VkFramebuffer> m_framebuffers;
};
}        // namespace Ilum::pass