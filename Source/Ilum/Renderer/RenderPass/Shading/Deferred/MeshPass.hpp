#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

#include <glm/glm.hpp>

namespace Ilum::pass
{
class MeshPass : public TRenderPass<MeshPass>
{
  public:
	MeshPass();

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	int32_t m_primitive_count = 1;

	Buffer m_debug_buffer;
	std::array<uint32_t, 10000> m_debug_data;
};
}        // namespace Ilum::pass