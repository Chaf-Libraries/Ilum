#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

#include <glm/glm.hpp>

namespace Ilum::pass
{
class CullingPass : public TRenderPass<CullingPass>
{
  public:
	CullingPass();

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

  private:
	scope<Buffer> m_count_buffer = nullptr;
	struct
	{
		glm::mat4 view;
		glm::mat4 last_view;
		float P00;
		float P11;
		float znear;
		float zfar;
		float zbuffer_width;
		float zbuffer_height;
		uint32_t draw_count;
		uint32_t frustum_enable;
		uint32_t backface_enable;
		uint32_t occlusion_enable;
	}m_cull_data;
};
}        // namespace Ilum::pass