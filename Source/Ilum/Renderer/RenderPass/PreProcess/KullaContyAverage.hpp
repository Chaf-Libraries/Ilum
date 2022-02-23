#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class KullaContyAverage : public TRenderPass<KullaContyAverage>
{
  public:
	KullaContyAverage();

	~KullaContyAverage() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	const uint32_t Resolution = 128;
	bool           m_finish   = false;
	Image          m_kulla_conty_average;
};
}        // namespace Ilum::pass