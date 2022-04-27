#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
// Extract bright part for blooming
class Tonemapping : public TRenderPass<Tonemapping>
{
  public:
	Tonemapping(const std::string &input, const std::string &output);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::string m_input;
	std::string m_output;

	struct
	{
		float   brightness = 1.f;
		float   contrast   = 1.f;
		float   saturation = 1.f;
		float   vignette   = 0.f;
		float   avgLum     = 1.f;
		int32_t autoExposure = 0;
		float   Ywhite = 0.5f;        // Burning white
		float   key    = 0.5f;        // Log-average luminance
	} m_push_data;
};
}        // namespace Ilum::pass