#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

#define FXAA_QUALITY_LOW "FXAA_QUALITY_LOW"
#define FXAA_QUALITY_MEDIUM "FXAA_QUALITY_MEDIUM"
#define FXAA_QUALITY_HIGH "FXAA_QUALITY_HIGH"


namespace Ilum::pass
{
// Fast approXimate Anti-Aliasing
class FXAA : public TRenderPass<FXAA>
{
  public:
	// Input should have alpha channel for luminance
	FXAA(const std::string &input, const std::string &output, const std::string &quality = FXAA_QUALITY_HIGH);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::string m_input;
	std::string m_output;
	std::string m_quality;

	struct
	{
		float fixed_threshold = 0.8333f;
		float relative_threshold = 0.166f;
		float subpixel_blending  = 0.7f;
	}m_push_constants;
};
}        // namespace Ilum::pass