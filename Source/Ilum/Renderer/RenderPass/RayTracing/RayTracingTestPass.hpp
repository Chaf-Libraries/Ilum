#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class RayTracingTestPass : public TRenderPass<RayTracingTestPass>
{
  public:
	RayTracingTestPass();

	~RayTracingTestPass() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	Image m_render_result;
};
}        // namespace Ilum::pass