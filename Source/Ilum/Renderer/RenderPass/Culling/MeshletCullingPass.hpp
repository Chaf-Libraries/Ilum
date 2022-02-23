#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

#include <glm/glm.hpp>

namespace Ilum::pass
{
class MeshletCullingPass : public TRenderPass<MeshletCullingPass>
{
  public:
	MeshletCullingPass() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	struct
	{
		uint32_t enable_frustum_culling   = 0;
		uint32_t enable_backface_culling  = 0;
		uint32_t enable_occlusion_culling = 0;
	}m_culling_mode;

	bool enable_culling = false;
};
}        // namespace Ilum::pass