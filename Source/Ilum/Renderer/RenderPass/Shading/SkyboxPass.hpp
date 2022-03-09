#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class SkyboxPass : public TRenderPass<SkyboxPass>
{
  public:
	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	enum class EnvLightType
	{
		None,
		HDR,
		// Atmospheric
	};

	EnvLightType m_type = EnvLightType::None;
};
}        // namespace Ilum::pass