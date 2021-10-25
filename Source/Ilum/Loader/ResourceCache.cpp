#include "ResourceCache.hpp"

#include "Loader/ImageLoader/ImageLoader.hpp"
#include "Loader/ModelLoader/ModelLoader.hpp"

#include "Device/LogicalDevice.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Pipeline/ShaderCompiler.hpp"

#include "File/FileSystem.hpp"

namespace Ilum
{
ImageReference ResourceCache::loadImage(const std::string &filepath)
{
	if (m_image_map.find(filepath) != m_image_map.end())
	{
		return m_image_cache.at(m_image_map.at(filepath));
	}

	m_image_map[filepath] = m_image_cache.size();
	m_image_cache.emplace_back(Image());
	ImageLoader::loadImageFromFile(m_image_cache.back(), filepath);

	return m_image_cache.back();
}

bool ResourceCache::hasImage(const std::string &filepath)
{
	return m_image_map.find(filepath) != m_image_map.end();
}

const std::unordered_map<std::string, size_t> &ResourceCache::getImages() const
{
	return m_image_map;
}

ModelReference ResourceCache::loadModel(const std::string &filepath)
{
	if (m_model_map.find(filepath) != m_model_map.end())
	{
		return m_model_cache.at(m_model_map.at(filepath));
	}

	m_model_map[filepath] = m_model_cache.size();
	m_model_cache.emplace_back(Model());
	ModelLoader::load(m_model_cache.back(), filepath);

	return m_model_cache.back();
}

bool ResourceCache::hasModel(const std::string &filepath)
{
	return m_model_map.find(filepath) != m_model_map.end();
}

const std::unordered_map<std::string, size_t> &ResourceCache::getModels() const
{
	return m_model_map;
}
}        // namespace Ilum