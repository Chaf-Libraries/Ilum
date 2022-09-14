#pragma once

#include "ResourceMeta.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace Ilum
{
class RHIContext;

class ResourceManager
{
  public:
	ResourceManager(RHIContext *rhi_context);

	~ResourceManager();

	void ImportTexture(const std::string &filename);

	void ImportModel(const std::string &filename);

	void AddSceneMeta(SceneMeta &&meta);

	void AddRenderGraphMeta(RenderGraphMeta &&meta);

	const std::vector<std::unique_ptr<TextureMeta>> &GetTextureMeta() const;

	const std::vector<std::unique_ptr<ModelMeta>> &GetModelMeta() const;

	const std::unordered_map<std::string, std::unique_ptr<SceneMeta>> &GetSceneMeta() const;

	const std::unordered_map<std::string, std::unique_ptr<RenderGraphMeta>> &GetRenderGraphMeta() const;

	const std::vector<RHITexture *> &GetTextureArray() const;

	const TextureMeta *GetTexture(const std::string &uuid);

	const ModelMeta *GetModel(const std::string &uuid);

	const SceneMeta *GetScene(const std::string &uuid);

	const RenderGraphMeta *GetRenderGraph(const std::string &uuid);

	RHITexture *GetThumbnail(ResourceType type) const;

  private:
	RHIContext *p_rhi_context = nullptr;

	// Texture
	std::vector<RHITexture *> m_texture_array;

	std::vector<std::unique_ptr<TextureMeta>> m_textures;
	std::unordered_map<std::string, size_t>   m_texture_index;

	// Model
	std::vector<std::unique_ptr<ModelMeta>> m_models;
	std::unordered_map<std::string, size_t> m_model_index;

	// Scene
	std::unordered_map<std::string, std::unique_ptr<SceneMeta>> m_scenes;

	// Render Graph
	std::unordered_map<std::string, std::unique_ptr<RenderGraphMeta>> m_render_graphs;

	std::unordered_map<ResourceType, std::unique_ptr<RHITexture>> m_thumbnails;

	const TextureDesc ThumbnailDesc = {"", 48, 48, 1, 1, 1, 1, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::Transfer | RHITextureUsage::ShaderResource};
};
}        // namespace Ilum