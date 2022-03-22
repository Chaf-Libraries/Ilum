#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class BRDFPreIntegrate : public TRenderPass<BRDFPreIntegrate>
{
  public:
	BRDFPreIntegrate() = default;

	~BRDFPreIntegrate() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	bool m_finish = true;
};
}        // namespace Ilum::pass