#pragma once

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class ImGuiContext
{
  public:
	ImGuiContext(RHIContext *context, Window* window);

	~ImGuiContext();

	void NewFrame();

	void Render();

  private:
	void UpdateBuffers();

	void SetStyle();

	void InitializePlatformInterface();

  private:
	RHIContext *p_context = nullptr;
	Window     *p_window  = nullptr;

	std::unique_ptr<RHIPipelineState> m_pipeline_state = nullptr;
	std::unique_ptr<RHIDescriptor>    m_descriptor     = nullptr;

	std::unique_ptr<RHIBuffer> m_vertex_buffer  = nullptr;
	std::unique_ptr<RHIBuffer> m_index_buffer   = nullptr;
	std::unique_ptr<RHIBuffer> m_uniform_buffer = nullptr;

	std::unique_ptr<RHISampler> m_sampler = nullptr;

	std::unique_ptr<RHIShader> m_vertex_shader   = nullptr;
	std::unique_ptr<RHIShader> m_fragment_shader = nullptr;

	std::unique_ptr<RHIRenderTarget> m_render_target = nullptr;

	std::unique_ptr<RHITexture> m_font_atlas = nullptr;

	size_t m_vertex_count = 0;
	size_t m_index_count  = 0;

	struct
	{
		glm::vec2 scale;
		glm::vec2 translate;
	}m_constant_block;
};
}        // namespace Ilum