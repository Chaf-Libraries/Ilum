#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

#include <glm/glm.hpp>

#define USE_JITTER "USE_JITTER"

namespace Ilum::pass
{
class MeshPass : public TRenderPass<MeshPass>
{
  public:
	MeshPass(bool jitter = false);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	bool m_jitter = false;
};
}        // namespace Ilum::pass