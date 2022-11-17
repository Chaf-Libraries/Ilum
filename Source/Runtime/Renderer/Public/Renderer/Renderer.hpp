#pragma once

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class Scene;
class ResourceManager;
class MaterialGraph;
class RenderGraph;

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

class Renderer
{
  public:
	Renderer(RHIContext *rhi_context, Scene *scene, ResourceManager *resource_manager);

	~Renderer();

	void Tick();

	void SetRenderGraph(std::unique_ptr<RenderGraph> &&render_graph);

	RenderGraph *GetRenderGraph() const;

	RHIContext *GetRHIContext() const;

	ResourceManager *GetResourceManager() const;

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

	RHIShader *RequireMaterialShader(MaterialGraph *material_graph, const std::string &filename, const std::string &entry_point, RHIShaderStage stage, std::vector<std::string> &&macros = {}, std::vector<std::string> &&includes = {});

  private:
	void UpdateScene();

  private:
	RHIContext *p_rhi_context = nullptr;

	Scene *p_scene = nullptr;

	ResourceManager *p_resource_manager = nullptr;

	glm::vec2 m_viewport = {};

	RHITexture *m_present_texture = nullptr;

	std::unique_ptr<RenderGraph> m_render_graph = nullptr;

	std::unordered_map<size_t, std::unique_ptr<RHIShader>> m_shader_cache;

	std::unordered_map<RHIShader *, ShaderMeta> m_shader_meta_cache;

	std::map<DummyTexture, std::unique_ptr<RHITexture>> m_dummy_textures;

	std::unique_ptr<RHIBuffer> m_view_buffer = nullptr;

	std::unique_ptr<RHIAccelerationStructure> m_tlas = nullptr;

	SceneInfo m_scene_info;
};
}        // namespace Ilum