#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class CubemapPrefilter : public TRenderPass<CubemapPrefilter>
{
  public:
	CubemapPrefilter();

	~CubemapPrefilter() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;
};
}        // namespace Ilum::pass