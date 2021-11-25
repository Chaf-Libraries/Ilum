#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class CullingPass : public TRenderPass<CullingPass>
{
  public:
	CullingPass();

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

  private:
	scope<Buffer> m_buffer;
};
}        // namespace Ilum::pass