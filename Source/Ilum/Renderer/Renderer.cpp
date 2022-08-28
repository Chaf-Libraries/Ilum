#include "Renderer.hpp"

namespace Ilum
{
Renderer::Renderer(RHIContext *rhi_context)
{
}

Renderer::~Renderer()
{
}

void Renderer::Tick()
{
}

void Renderer::SetRenderGraph(std::unique_ptr<RenderGraph> &&render_graph)
{
	m_render_graph = std::move(render_graph);
}

RenderGraph *Renderer::GetRenderGraph() const
{
	return m_render_graph.get();
}
}        // namespace Ilum