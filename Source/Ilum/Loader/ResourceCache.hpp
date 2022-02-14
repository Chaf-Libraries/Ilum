#pragma once

#include <Graphics/Resource/Image.hpp>
#include "Graphics/Model/Model.hpp"
#include "Graphics/Shader/Shader.hpp"
#include "Graphics/Shader/ShaderReflection.hpp"

namespace Ilum
{
class ResourceCache
{
  public:
	ResourceCache() = default;

	~ResourceCache() = default;

	Graphics::ImageReference loadImage(const std::string &filepath);

	void loadImageAsync(const std::string &filepath);

	void removeImage(const std::string &filepath);

	bool hasImage(const std::string &filepath) const;

	const std::unordered_map<std::string, size_t> &getImages();

	void updateImageReferences();

	const std::vector<Graphics::ImageReference> &getImageReferences() const;

	uint32_t imageID(const std::string &filepath);

	void clearImages();

	bool isLoadingImage() const;

	ModelReference loadModel(const std::string &name);

	void loadModelAsync(const std::string &filepath);

	void removeModel(const std::string &filepath);

	bool hasModel(const std::string &filepath);

	const std::unordered_map<std::string, size_t> &getModels();

	bool isLoadingModel() const;

	void clearModels();

	const uint32_t getVerticesCount() const;

	const uint32_t getIndicesCount() const;

	void clear();

	void flush();

  private:
	// Cache image
	std::vector<Graphics::Image>                                  m_image_cache;
	std::unordered_map<std::string, size_t>             m_image_map;
	std::vector<std::string>                            m_deprecated_image;
	std::unordered_set<std::string>                     m_new_image;
	std::unordered_map<std::string, std::future<Graphics::Image>> m_image_futures;
	std::vector<Graphics::ImageReference>                         m_image_references;

	// Cache model
	std::vector<Model>                                  m_model_cache;
	std::unordered_map<std::string, size_t>             m_model_map;
	std::vector<std::string>                            m_deprecated_model;
	std::unordered_set<std::string>                     m_new_model;
	std::unordered_map<std::string, std::future<Model>> m_model_futures;

	uint32_t m_vertices_count = 0;
	uint32_t m_indices_count  = 0;

	std::mutex m_image_mutex;
	std::mutex m_model_mutex;
};
}        // namespace Ilum