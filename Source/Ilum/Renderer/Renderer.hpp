#pragma once

#include <RHI/RHIContext.hpp>
#include <RenderCore/RenderGraph/RenderGraph.hpp>

namespace Ilum
{
class Scene;

class Renderer
{
  public:
	Renderer(RHIContext *rhi_context);

	~Renderer();

	void Tick();

	void SetRenderGraph(std::unique_ptr<RenderGraph> &&render_graph);

	RenderGraph *GetRenderGraph() const;

	RHIContext *GetRHIContext() const;

	void SetViewport(float width, float height);

	glm::vec2 GetViewport() const;

	void SetPresentTexture(RHITexture *present_texture);

	RHITexture *GetPresentTexture() const;

	void SetScene(std::unique_ptr<Scene> &&scene);

	Scene *GetScene() const;

	void Reset();

  public:
	// Shader utils
	RHIShader *RequireShader(const std::string &filename, const std::string &entry_point, RHIShaderStage stage, const std::vector<std::string> &macros = {});

	ShaderMeta RequireShaderMeta(RHIShader *shader) const;

  private:
	RHIContext *p_rhi_context = nullptr;

	std::unique_ptr<Scene> p_scene = nullptr;

	glm::vec2 m_viewport = {};

	RHITexture *m_present_texture = nullptr;

	std::unique_ptr<RenderGraph> m_render_graph = nullptr;

	std::unordered_map<size_t, std::unique_ptr<RHIShader>> m_shader_cache;

	std::unordered_map<RHIShader *, ShaderMeta> m_shader_meta_cache;
};
}        // namespace Ilum