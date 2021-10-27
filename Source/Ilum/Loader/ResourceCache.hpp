#pragma once

#include "Graphics/Image/Image.hpp"
#include "Graphics/Model/Model.hpp"
#include "Graphics/Pipeline/Shader.hpp"
#include "Graphics/Pipeline/ShaderReflection.hpp"

namespace Ilum
{
class ResourceCache
{
  public:
	ResourceCache() = default;

	~ResourceCache() = default;

	ImageReference loadImage(const std::string &filepath);

	void loadImageAsync(const std::string &filepath);

	bool hasImage(const std::string &filepath);

	const std::unordered_map<std::string, size_t> &getImages() const;

	ModelReference loadModel(const std::string &filepath);

	void loadModelAsync(const std::string &filepath);

	bool hasModel(const std::string &filepath);

	const std::unordered_map<std::string, size_t> &getModels() const;

  private:
	// Cache image
	std::vector<Image>                      m_image_cache;
	std::unordered_map<std::string, size_t> m_image_map;

	// Cache model
	std::vector<Model>                      m_model_cache;
	std::unordered_map<std::string, size_t> m_model_map;

	std::mutex m_image_mutex;
	std::mutex m_model_mutex;
};
}        // namespace Ilum