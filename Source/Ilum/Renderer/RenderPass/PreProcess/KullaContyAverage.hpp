#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class KullaContyAverage : public TRenderPass<KullaContyAverage>
{
  public:
	KullaContyAverage() = default;

	~KullaContyAverage() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	bool           m_finish   = false;
};
}        // namespace Ilum::pass