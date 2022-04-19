#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class MeshViewPass : public TRenderPass<MeshViewPass>
{
  public:
	MeshViewPass() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::string m_texture    = "";
	uint32_t    m_texture_id = std::numeric_limits<uint32_t>::max();
	uint32_t    m_parameterization = 0;
};
}        // namespace Ilum::pass