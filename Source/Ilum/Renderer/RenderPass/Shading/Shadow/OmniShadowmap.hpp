#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

#include <glm/glm.hpp>

namespace Ilum::pass
{
class OmniShadowmapPass : public TRenderPass<OmniShadowmapPass>
{
  public:
	OmniShadowmapPass() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	struct
	{
		glm::mat4 view_projection = glm::mat4(1.f);
		uint32_t  light_id        = {};
		uint32_t  face_id         = {};
		float     depth_bias      = 0.01f;
	} m_push_block;

	int32_t m_light_id = 0;
	int32_t m_face_id  = 0;

	VkExtent2D m_resolution = {2048, 2048};
};
}        // namespace Ilum::pass