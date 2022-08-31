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

	// Temp
	RHITexture *GetTexture();

  private:
	RHIContext                  *p_rhi_context  = nullptr;
	std::unique_ptr<RenderGraph> m_render_graph = nullptr;

	std::unique_ptr<RHIShader>        m_shader         = nullptr;
	std::unique_ptr<RHIDescriptor>    m_descriptor     = nullptr;
	std::unique_ptr<RHIPipelineState> m_pipeline_state = nullptr;
	std::unique_ptr<RHITexture>       m_texture        = nullptr;
	std::unique_ptr<RHIBuffer>        m_buffer         = nullptr;
};
}        // namespace Ilum