#pragma once

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class Scene;
class ResourceManager;
class MaterialGraph;
class RenderGraph;
class RenderGraphBlackboard;
class ShaderBuilder;

namespace Cmpt
{
class Camera;
}

class EXPORT_API Renderer
{
  public:
	Renderer(RHIContext *rhi_context, Scene *scene, ResourceManager *resource_manager);

	~Renderer();

	void Tick();

	void SetRenderGraph(std::unique_ptr<RenderGraph> &&render_graph);

	RenderGraph *GetRenderGraph() const;

	RHIContext *GetRHIContext() const;

	ResourceManager *GetResourceManager() const;

	RenderGraphBlackboard &GetRenderGraphBlackboard();

	void SetViewport(float width, float height);

	glm::vec2 GetViewport() const;

	void SetAnimationTime(float time);

	void SetPresentTexture(RHITexture *present_texture);

	RHITexture *GetPresentTexture() const;

	float GetMaxAnimationTime() const;

	void UpdateView(Cmpt::Camera *camera);

	Scene *GetScene() const;

	void Reset();

  public:
	// Shader utils
	RHIShader *RequireShader(const std::string &filename, const std::string &entry_point, RHIShaderStage stage, std::vector<std::string> &&macros = {}, std::vector<std::string> &&includes = {}, bool cuda = false, bool force_recompile = false);

	ShaderMeta RequireShaderMeta(RHIShader *shader) const;

	// RHIShader *RequireMaterialShader(MaterialGraph *material_graph, const std::string &filename, const std::string &entry_point, RHIShaderStage stage, std::vector<std::string> &&macros = {}, std::vector<std::string> &&includes = {});

  private:
	void UpdateGPUScene();

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum