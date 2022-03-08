#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

#include <glm/glm.hpp>

namespace Ilum::pass
{
class GeometryPass : public TRenderPass<GeometryPass>
{
  public:
	GeometryPass();

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

  private:
	enum class RenderMode
	{
		Polygon,
		WireFrame,
		PointCloud
	};

	struct
	{
		glm::mat4 transform =glm::mat4(1.f);
		uint32_t  dynamic   = 0;
	} m_vertex_block;

	RenderMode m_render_mode = RenderMode::Polygon;
};
}        // namespace Ilum::pass