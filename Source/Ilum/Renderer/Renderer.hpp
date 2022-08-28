#pragma once

#include <RHI/RHIContext.hpp>

#include <RenderCore/RenderGraph/RenderGraph.hpp>

namespace Ilum
{
class Renderer
{
  public:
	Renderer(RHIContext *rhi_context);

	~Renderer();

	void Tick();

	void SetRenderGraph(std::unique_ptr<RenderGraph> &&render_graph);

	RenderGraph *GetRenderGraph() const;

  private:
	RHIContext *p_rhi_context = nullptr;
	std::unique_ptr<RenderGraph> m_render_graph = nullptr;
};
}        // namespace Ilum