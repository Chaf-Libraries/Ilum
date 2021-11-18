#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

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
	std::vector<scope<CommandBuffer>> m_secondary_command_buffers;
};
}        // namespace Ilum::pass