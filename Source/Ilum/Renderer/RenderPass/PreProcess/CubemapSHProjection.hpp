#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class CubemapSHProjection : public TRenderPass<CubemapSHProjection>
{
  public:
	CubemapSHProjection();

	~CubemapSHProjection() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	int32_t m_face_id = 0;
};
}        // namespace Ilum::pass