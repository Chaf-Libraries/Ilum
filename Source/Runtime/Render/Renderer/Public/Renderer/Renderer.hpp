#pragma once

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class Scene;
class ResourceManager;
class MaterialGraph;
class RenderGraph;
class ShaderBuilder;

enum class DummyTexture
{
	WhiteOpaque,
	BlackOpaque,
	WhiteTransparent,
	BlackTransparent
};

struct ViewInfo
{
	glm::mat4 view_matrix;
	glm::mat4 inv_view_matrix;
	glm::mat4 projection_matrix;
	glm::mat4 inv_projection_matrix;
	glm::mat4 view_projection_matrix;
	glm::vec3 position;
	uint32_t  frame_count;
};

struct SceneInfo
{
	std::vector<RHITexture *>    textures;
	std::vector<MaterialGraph *> materials;

	std::unique_ptr<RHIBuffer> light_buffer = nullptr;

	std::vector<RHIBuffer *> static_vertex_buffers;
	std::vector<RHIBuffer *> static_index_buffers;
	std::vector<RHIBuffer *> meshlet_vertex_buffers;
	std::vector<RHIBuffer *> meshlet_primitive_buffers;
	std::vector<RHIBuffer *> meshlet_buffers;

	std::unique_ptr<RHIBuffer> instance_buffer = nullptr;

	RHIAccelerationStructure *top_level_as = nullptr;

	std::vector<uint32_t> meshlet_count;
};

class EXPORT_API Renderer
{
  public:
	Renderer(RHIContext *rhi_context, Scene *scene);

	~Renderer();

	void Tick();

	void SetRenderGraph(std::unique_ptr<RenderGraph> &&render_graph);

	RenderGraph *GetRenderGraph() const;

	RHIContext *GetRHIContext() const;

	void SetViewport(float width, float height);

	glm::vec2 GetViewport() const;

	void SetPresentTexture(RHITexture *present_texture);

	RHITexture *GetPresentTexture() const;

	void SetViewInfo(const ViewInfo &view_info);

	RHIBuffer *GetViewBuffer() const;

	Scene *GetScene() const;

	void Reset();

	RHITexture *GetDummyTexture(DummyTexture dummy) const;

	RHIAccelerationStructure *GetTLAS() const;

	void DrawScene(RHICommand *cmd_buffer, RHIPipelineState *pipeline_state, RHIDescriptor *descriptor, bool mesh_shader);

	const SceneInfo &GetSceneInfo() const;

  public:
	// Shader utils
	RHIShader *RequireShader(const std::string &filename, const std::string &entry_point, RHIShaderStage stage, std::vector<std::string> &&macros = {}, std::vector<std::string> &&includes = {}, bool cuda = false, bool force_recompile = false);

	ShaderMeta RequireShaderMeta(RHIShader *shader) const;

	// RHIShader *RequireMaterialShader(MaterialGraph *material_graph, const std::string &filename, const std::string &entry_point, RHIShaderStage stage, std::vector<std::string> &&macros = {}, std::vector<std::string> &&includes = {});

  private:
	void UpdateScene();

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum