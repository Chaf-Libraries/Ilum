#pragma once

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class GuiContext
{
  public:
	GuiContext(RHIContext *context, Window *window);

	~GuiContext();

	void BeginFrame();

	void EndFrame();

	void Render();

  private:
	void SetStyle();

	void InitializePlatformInterface();

  private:
	RHIContext *p_context = nullptr;
	Window     *p_window  = nullptr;

	std::unique_ptr<RHIPipelineState> m_pipeline_state = nullptr;
	std::unique_ptr<RHIDescriptor>    m_descriptor     = nullptr;

	RHISampler* m_sampler = nullptr;

	std::unique_ptr<RHIShader> m_vertex_shader   = nullptr;
	std::unique_ptr<RHIShader> m_fragment_shader = nullptr;

	std::unique_ptr<RHIRenderTarget> m_render_target = nullptr;

	std::unique_ptr<RHITexture> m_font_atlas = nullptr;

	size_t m_vertex_count = 0;
	size_t m_index_count  = 0;
};
}        // namespace Ilum